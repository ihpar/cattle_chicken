import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path


class TrainerReg:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        lr=1e-3,
        epochs=200,
        patience=20,
        device="cpu",
        save_path="best_reg_model.pt"
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.save_path = Path(save_path)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):

            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(self.model.state_dict(), self.save_path)
                print("Best model saved.")

            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Best validation loss: {best_val_loss:.4f}")

    def train_epoch(self):

        self.model.train()
        total_loss = 0

        for x, y in self.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(self.train_loader)

    def validate(self):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            for x, y in self.val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x.size(0)

        return total_loss / len(self.val_loader)

    def test(self):

        print("\nTesting best model...")

        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        test_loss = 0.0

        all_preds, all_targets = [], []

        with torch.no_grad():

            for x, y in self.test_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)

                loss = self.criterion(outputs, y)
                test_loss += loss.item() * x.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(y.cpu())

        test_loss /= len(self.test_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"Test MSE: {test_loss:.6f}")
        print(f"Test MAE: {torch.mean(torch.abs(all_preds - all_targets)):.6f}")

        return test_loss, all_preds, all_targets
