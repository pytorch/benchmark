import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, debug: str = None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)
        self.model = torch.compile(self.model)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if 0 and with_cuda and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        self.debug = debug

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        data_iter = enumerate(data_loader)
        import time

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        self.log_freq = 10
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            # [print(key, value.shape) for key, value in data.items()]
            # print(data)
            st, ed= 0,0
            for j in range(100):
                if j==50:
                    st = time.time()
                
                with torch.autocast('cuda'):
                    # 1. forward the next_sentence_prediction and masked_lm model
                    next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

                    # 2-1. NLL(negative log likelihood) loss of is_next classification result
                    next_loss = self.criterion(next_sent_output, data["is_next"])

                    # 2-2. NLLLoss of predicting masked token word
                    mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

                    # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
                    loss = next_loss + mask_loss

                    # 3. backward and optimization only in train
                    if train:
                        self.optim_schedule.zero_grad()
                        loss.backward()
                        self.optim_schedule.step_and_update_lr()

                    # next sentence prediction accuracy
                    correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
                    avg_loss += loss.item()
                    total_correct += correct
                    total_element += data["is_next"].nelement()

                    post_fix = {
                        "epoch": epoch,
                        "iter": j,
                        "avg_loss": avg_loss / (j + 1),
                        "avg_acc": total_correct / total_element * 100,
                        "loss": loss.item()
                    }

                    if j % self.log_freq == 0:
                        print(str(post_fix))
                    if j == 99:
                        ed = time.time()
                        cost = ed - st
                        shape = data['bert_input'].shape
                        print("ips: {} token/s, cost: {} sec".format(shape[0]*shape[1]*50/cost, cost))
                        exit()

            # if self.debug and epoch == 1 and i == 0:
            #     torch.save(next_sent_output, self.debug)

        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
        #       total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
