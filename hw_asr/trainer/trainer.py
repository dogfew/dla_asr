import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker
from torch.cuda.amp import GradScaler
from hw_asr.utils import optional_autocast


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        text_encoder,
        log_step=200,  # how often WANDB will log
        log_predictions_step_epoch=5,
        mixed_precision=True,
        do_beam_search=False,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.log_predictions_step_epoch = log_predictions_step_epoch
        self.do_beam_search = do_beam_search
        self.mixed_precision = mixed_precision
        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

        self.scaler = GradScaler(enabled=self.mixed_precision)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                # self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                if epoch % self.log_predictions_step_epoch == 0:
                    self._log_predictions(**batch, is_train=True)
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        with optional_autocast(self.mixed_precision):
            outputs = self.model(**batch)
            if type(outputs) is dict:
                batch.update(outputs)
            else:
                batch["logits"] = outputs

            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = self.model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["loss"] = self.criterion(**batch)
        if is_train:
            self.scaler.scale(batch["loss"]).backward()
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            if epoch % self.log_predictions_step_epoch == 0:
                self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
        self,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        audio,
        examples_to_log=10,
        is_train=False,
        *args,
        **kwargs,
    ):
        if self.writer is None:
            return
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        if self.do_beam_search:
            probs_cpu = torch.exp(log_probs).cpu().detach().numpy()
            probs_length_numpy = log_probs_length.numpy()
            lm_beam_search_texts = [
                self.text_encoder.lm_ctc_beam_search(
                    probs=probs_cpu[i], probs_length=probs_length_numpy[i]
                )[0].text
                for i in range(examples_to_log)
            ]
            beam_search_texts = [
                self.text_encoder.ctc_beam_search(
                    probs=probs_cpu[i], probs_length=probs_length_numpy[i]
                )[0].text
                for i in range(examples_to_log)
            ]
        else:
            lm_beam_search_texts = ["" for _ in range(examples_to_log)]
            beam_search_texts = ["" for _ in range(examples_to_log)]
        tuples = [
            argmax_texts,
            text,
            lm_beam_search_texts,
            beam_search_texts,
            argmax_texts_raw,
            audio_path,
            audio,
        ]
        tuples = [i[:examples_to_log] for i in tuples]
        tuples = list(zip(*tuples))
        shuffle(tuples)
        rows = {}
        for (
            pred,
            target,
            lm_beam_search_pred,
            beam_search_pred,
            raw_pred,
            audio_path,
            audio_,
        ) in tuples:
            target = BaseTextEncoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "audio": wandb.Audio(audio_path),
                "augmented_audio": wandb.Audio(
                    audio_.squeeze(), sample_rate=self.config["preprocessing"]["sr"]
                ),
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
            }
            if not is_train:
                rows[Path(audio_path).name].pop("augmented_audio")
            if self.do_beam_search:
                wer_beam_search = calc_wer(target, lm_beam_search_pred) * 100
                cer_beam_search = calc_cer(target, lm_beam_search_pred) * 100

                rows[Path(audio_path).name].update(
                    {
                        "predictions_lm_beam_search": lm_beam_search_pred,
                        "predictions_beam_search": beam_search_pred,
                        "wer_lm_beam_search": wer_beam_search,
                        "cer_lm_beam_search": cer_beam_search,
                    }
                )
            rows[Path(audio_path).name].update({"wer": wer, "cer": cer})
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        # nan occurs in first batch in first run with grad scaler
                        torch.nan_to_num(p.grad, nan=0).detach(),
                        norm_type,
                    ).cpu()
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
