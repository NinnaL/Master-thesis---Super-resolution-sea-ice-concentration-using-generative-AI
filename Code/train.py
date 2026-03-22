import argparse
import json
import math
import os
import time
import torch
from typing import List

from lib.dataset.dataloader import (
	DEFAULT_DATA_DIRS,
	DEFAULT_TRAINING_INDEX_CSV,
	create_train_val_dataloaders,
)
from lib.model.Baseline import EncDec


def _num_parameters(model: torch.nn.Module) -> int:
	return sum(p.numel() for p in model.parameters())


def _run_epoch(
	model: torch.nn.Module,
	loader,
	device: torch.device,
	criterion: torch.nn.Module,
	optimizer=None,
):
	training = optimizer is not None
	if training:
		model.train()
	else:
		model.eval()

	sum_sq = 0.0
	sum_abs = 0.0
	count = 0
	batch_count = 0

	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for batch in loader:
			if batch is None:
				continue

			x, y = batch
			x = x.to(device)
			y = y.to(device)

			if training:
				optimizer.zero_grad(set_to_none=True)

			pred = model(x)
			loss = criterion(pred, y)

			if training:
				loss.backward()
				optimizer.step()

			err = pred - y
			sum_sq += torch.sum(err * err).item()
			sum_abs += torch.sum(torch.abs(err)).item()
			count += err.numel()
			batch_count += 1

	if count == 0:
		return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "batches": 0}

	mse = sum_sq / count
	mae = sum_abs / count
	rmse = math.sqrt(mse)
	return {"mse": mse, "mae": mae, "rmse": rmse, "batches": batch_count}


def train(
	training_index_csv: str,
	data_dirs: List[str],
	epochs: int,
	batch_size: int,
	lr: float,
	val_split: float,
	output_dir: str,
	image_size: int,
) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	os.makedirs(output_dir, exist_ok=True)
	metrics_csv_path = os.path.join(output_dir, "metrics.csv")
	summary_json_path = os.path.join(output_dir, "training_summary.json")

	train_loader, val_loader, dataset = create_train_val_dataloaders(
		training_index_csv=training_index_csv,
		data_dirs=data_dirs,
		batch_size=batch_size,
		val_split=val_split,
		image_size=image_size,
		num_workers=0,
		shuffle=True,
	)

	model = EncDec().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.MSELoss()

	train_samples = len(train_loader.dataset)
	val_samples = len(val_loader.dataset)
	model_params = _num_parameters(model)

	print("Training configuration:")
	print(f"  Device            : {device}")
	print(f"  Training index    : {training_index_csv}")
	print(f"  Data directories  : {data_dirs}")
	print(f"  Total samples     : {len(dataset)}")
	print(f"  Train samples     : {train_samples}")
	print(f"  Validation samples: {val_samples}")
	print(f"  Epochs            : {epochs}")
	print(f"  Batch size        : {batch_size}")
	print(f"  Learning rate     : {lr}")
	print(f"  Image size        : {image_size}")
	print(f"  Model parameters  : {model_params}")

	history = []
	start_time = time.time()

	best_val = float("inf")
	best_epoch = -1
	for epoch in range(1, epochs + 1):
		epoch_start = time.time()
		train_metrics = _run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
		val_metrics = _run_epoch(model, val_loader, device, criterion, optimizer=None)
		epoch_seconds = time.time() - epoch_start

		record = {
			"epoch": epoch,
			"train_mse": train_metrics["mse"],
			"train_mae": train_metrics["mae"],
			"train_rmse": train_metrics["rmse"],
			"val_mse": val_metrics["mse"],
			"val_mae": val_metrics["mae"],
			"val_rmse": val_metrics["rmse"],
			"train_batches": train_metrics["batches"],
			"val_batches": val_metrics["batches"],
			"epoch_seconds": epoch_seconds,
		}
		history.append(record)

		print(
			f"Epoch {epoch:03d} | "
			f"train_mse={record['train_mse']:.6f} train_mae={record['train_mae']:.6f} train_rmse={record['train_rmse']:.6f} | "
			f"val_mse={record['val_mse']:.6f} val_mae={record['val_mae']:.6f} val_rmse={record['val_rmse']:.6f} | "
			f"time={epoch_seconds:.1f}s"
		)

		if record["val_mse"] < best_val:
			best_val = record["val_mse"]
			best_epoch = epoch
			ckpt_path = os.path.join(output_dir, "best_model.pt")
			torch.save(
				{
					"epoch": epoch,
					"model_state": model.state_dict(),
					"optimizer_state": optimizer.state_dict(),
					"val_mse": record["val_mse"],
					"val_mae": record["val_mae"],
					"val_rmse": record["val_rmse"],
				},
				ckpt_path,
			)

	final_path = os.path.join(output_dir, "last_model.pt")
	torch.save(model.state_dict(), final_path)

	with open(metrics_csv_path, "w", encoding="utf-8") as f:
		f.write("epoch,train_mse,train_mae,train_rmse,val_mse,val_mae,val_rmse,train_batches,val_batches,epoch_seconds\n")
		for row in history:
			f.write(
				f"{row['epoch']},{row['train_mse']},{row['train_mae']},{row['train_rmse']},"
				f"{row['val_mse']},{row['val_mae']},{row['val_rmse']},"
				f"{row['train_batches']},{row['val_batches']},{row['epoch_seconds']}\n"
			)

	total_seconds = time.time() - start_time
	best_row = min(history, key=lambda r: r["val_mse"]) if history else None
	summary = {
		"training_index_csv": training_index_csv,
		"data_dirs": data_dirs,
		"device": str(device),
		"epochs": epochs,
		"batch_size": batch_size,
		"learning_rate": lr,
		"image_size": image_size,
		"total_samples": len(dataset),
		"train_samples": train_samples,
		"val_samples": val_samples,
		"model_parameters": model_params,
		"total_seconds": total_seconds,
		"best_epoch": best_epoch,
		"best_val_mse": best_row["val_mse"] if best_row else None,
		"best_val_mae": best_row["val_mae"] if best_row else None,
		"best_val_rmse": best_row["val_rmse"] if best_row else None,
		"metrics_csv": metrics_csv_path,
	}
	with open(summary_json_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("Training summary:")
	print(f"  Best epoch    : {best_epoch}")
	print(f"  Best val MSE  : {summary['best_val_mse']:.6f}" if summary["best_val_mse"] is not None else "  Best val MSE  : n/a")
	print(f"  Best val MAE  : {summary['best_val_mae']:.6f}" if summary["best_val_mae"] is not None else "  Best val MAE  : n/a")
	print(f"  Best val RMSE : {summary['best_val_rmse']:.6f}" if summary["best_val_rmse"] is not None else "  Best val RMSE : n/a")
	print(f"  Total time    : {total_seconds:.1f}s")
	print(f"  Metrics CSV   : {metrics_csv_path}")
	print(f"  Summary JSON  : {summary_json_path}")
	print(f"Training finished. Saved best checkpoint in {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train EncDec using training_index.csv")
	parser.add_argument(
		"--training-index",
		type=str,
		default=DEFAULT_TRAINING_INDEX_CSV,
	)
	parser.add_argument(
		"--data-dirs",
		type=str,
		nargs="+",
		default=DEFAULT_DATA_DIRS,
		help="One or more directories to recursively search for .nc and *_SIC.tiff files.",
	)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--val-split", type=float, default=0.1)
	parser.add_argument("--image-size", type=int, default=128)
	parser.add_argument("--output-dir", type=str, default="checkpoints")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(
		training_index_csv=args.training_index,
		data_dirs=args.data_dirs,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		val_split=args.val_split,
		output_dir=args.output_dir,
		image_size=args.image_size,
	)


