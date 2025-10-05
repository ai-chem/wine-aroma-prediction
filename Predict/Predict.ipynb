{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "912b0125-09ec-49f0-a1df-60d692b0a2e6",
   "metadata": {},
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "addf5c3d-5f6d-49fe-8b1f-3f144a95fb46",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8cc0f12-db56-4b3b-9fe1-6e23f8dbc290",
   "metadata": {},
   "source": [
    "## Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "39a9985b-a53b-4750-9a9d-859d14c556e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNet(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)\n",
    "        self.bn1   = nn.BatchNorm2d(64)\n",
    "        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)\n",
    "        self.bn2   = nn.BatchNorm2d(128)\n",
    "        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)\n",
    "\n",
    "        self.fc1   = nn.Linear(128 * 11 * 25, 512)\n",
    "        self.fc2   = nn.Linear(512, 128)\n",
    "        self.fc3   = nn.Linear(128, 10)\n",
    "        self.dropout = nn.Dropout(0.2)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))\n",
    "        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))\n",
    "        x = x.view(x.size(0), -1)               \n",
    "        x = F.leaky_relu(self.fc1(x))\n",
    "        x = self.dropout(x)\n",
    "        x = F.leaky_relu(self.fc2(x))\n",
    "        x = self.fc3(x)                      \n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "932ff65d-4c14-4d1f-abcd-9512e1e23351",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CustomDataset(torch.utils.data.Dataset):\n",
    "    def __init__(self, x, y):\n",
    "        self.x = x\n",
    "        self.y = y\n",
    "    def __len__(self):\n",
    "        return len(self.x)\n",
    "    def __getitem__(self, idx):\n",
    "        return self.x[idx], self.y[idx]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b64f1d0f-ea61-4774-9981-e1278beeb3e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_array(path_main: str):\n",
    "    if os.path.isfile(path_main):\n",
    "        return np.load(path_main, allow_pickle=False)\n",
    "    base, ext = os.path.splitext(path_main)\n",
    "    candidates = [path_main, base + \".npy\", base + \".np\"]\n",
    "    for p in candidates:\n",
    "        if os.path.isfile(p):\n",
    "            return np.load(p, allow_pickle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "89151367-a235-4d0e-a481-ba986111f17b",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_np = load_array(\"X_array.npy\")  \n",
    "Y_np = load_array(\"Y_array.npy\")\n",
    "\n",
    "X = torch.from_numpy(X_np).float()\n",
    "Y = torch.from_numpy(Y_np).float()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4ba5e4d-d6c1-43af-aa12-a550bb2adbf2",
   "metadata": {},
   "source": [
    "## Saved weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ae750c64-f036-4437-9814-b7888cb14e29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ConvNet(\n",
       "  (conv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "  (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "  (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)\n",
       "  (fc1): Linear(in_features=35200, out_features=512, bias=True)\n",
       "  (fc2): Linear(in_features=512, out_features=128, bias=True)\n",
       "  (fc3): Linear(in_features=128, out_features=10, bias=True)\n",
       "  (dropout): Dropout(p=0.2, inplace=False)\n",
       ")"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class_names = [\"Flowers\",\"Chocolate/Cacao\",\"Coffee\",\"Nuts\",\"Fruits\",\n",
    "               \"Citrus\",\"Berries\",\"Wood\",\"Tobacco/Smoke\",\"Herbs and spices\"]\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model = ConvNet().to(device)\n",
    "\n",
    "state = torch.load(\"cnn_weights.pt\", map_location=device)\n",
    "model.load_state_dict(state)\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d969f87a-0941-4c7d-a34a-8cc7fde18c7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_all = CustomDataset(X, Y)\n",
    "loader = torch.utils.data.DataLoader(dataset_all, batch_size=64, shuffle=False)\n",
    "\n",
    "all_probs = []\n",
    "all_labels = []\n",
    "with torch.no_grad():\n",
    "    for xb, yb in loader:\n",
    "        if xb.dim() == 3:              # (B,H,W) -> (B,1,H,W)\n",
    "            xb = xb.unsqueeze(1)\n",
    "        xb = xb.to(device).float()\n",
    "        yb = yb.to(device).float()\n",
    "\n",
    "        logits = model(xb)             # (B, C)\n",
    "        probs  = torch.sigmoid(logits).cpu().numpy()\n",
    "        all_probs.append(probs)\n",
    "        all_labels.append(yb.cpu().numpy())\n",
    "\n",
    "all_probs = np.concatenate(all_probs, axis=0)   # (N, C)\n",
    "all_labels = np.concatenate(all_labels, axis=0) # (N, C)\n",
    "\n",
    "pred_bin = (all_probs > 0.5).astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df29aabd-3d95-48d0-b77d-8e075243e683",
   "metadata": {},
   "source": [
    "## Choose a wine sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "fd0b6a76-6d05-47fd-aa37-6eb5dff831a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\\Sample #90\n",
      "Flowers            p=0.381  pred=0  true=1\n",
      "Chocolate/Cacao    p=0.209  pred=0  true=0\n",
      "Coffee             p=0.161  pred=0  true=0\n",
      "Nuts               p=0.634  pred=1  true=1\n",
      "Fruits             p=0.208  pred=0  true=0\n",
      "Citrus             p=0.691  pred=1  true=1\n",
      "Berries            p=0.075  pred=0  true=0\n",
      "Wood               p=0.007  pred=0  true=0\n",
      "Tobacco/Smoke      p=0.072  pred=0  true=0\n",
      "Herbs and spices   p=0.424  pred=0  true=0\n"
     ]
    }
   ],
   "source": [
    "idx = 90  # change index\n",
    "probs_1 = all_probs[idx].ravel()\n",
    "preds_1 = pred_bin[idx].ravel()\n",
    "true_1  = all_labels[idx].ravel()\n",
    "\n",
    "print(f\"\\Sample #{idx}\")\n",
    "for name, p, z, t in zip(class_names, probs_1, preds_1, true_1):\n",
    "    print(f\"{name:<18} p={p:.3f}  pred={int(z)}  true={int(t)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "b624e6ac-8b4e-43e5-87c8-bcdd1d37e165",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample accuracy: 0.9000  (9/10 correct labels)\n"
     ]
    }
   ],
   "source": [
    "acc_sample = (preds_1 == true_1).mean()\n",
    "n_correct  = int((preds_1 == true_1).sum())\n",
    "n_classes  = len(true_1)\n",
    "print(f\"Sample accuracy: {acc_sample:.4f}  ({n_correct}/{n_classes} correct labels)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2d1ff543-5929-43aa-9d2b-ca195fc3766d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# df = pd.DataFrame(all_probs, columns=[f\"p_{n}\" for n in class_names])\n",
    "# for i, n in enumerate(class_names):\n",
    "#     df[f\"pred_{n}\"] = pred_bin[:, i]\n",
    "#     df[f\"true_{n}\"] = all_labels[:, i].astype(int)\n",
    "# df.to_csv(\"cnn_predictions.csv\", index=False)\n",
    "# print(\"[saved] cnn_predictions.csv\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
