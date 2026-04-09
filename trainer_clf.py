import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

class TrainerClf:
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
        save_path="best_model.pt"
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.save_path = Path(save_path)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(self.epochs):

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                torch.save(self.model.state_dict(), self.save_path)
                print("Best model saved.")

            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Best validation accuracy: {best_val_acc:.4f}")

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / len(self.train_loader), correct / total

    def validate(self):

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in self.val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / len(self.val_loader), correct / total

    def test(self):

        print("\nTesting best model...")

        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for x, y in self.test_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}")

        return test_acc