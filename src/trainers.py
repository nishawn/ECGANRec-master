import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr, cal_mrr, precision_at_k


class Trainer:

    def __init__(self, generator_A,
                 generator_B,

                 discriminator_A,
                 discriminator_B,

                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.generator_A = generator_A
        self.discriminator_A = discriminator_A
        self.generator_B = generator_B
        self.discriminator_B = discriminator_B


        self.online_similarity_model = args.online_similarity_model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length * self.args.hidden_size, \
                                                  512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                        nn.Linear(512, self.args.hidden_size, bias=True))

        if self.cuda_condition:
            self.generator_A.cuda()
            self.discriminator_A.cuda()
            self.generator_B.cuda()
            self.discriminator_B.cuda()


        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        # self.optim = Adam(list(self.generator.parameters())+list(self.discriminator.parameters()), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim_A = Adam(list(self.generator_A.parameters()) + list(self.discriminator_A.parameters()),
                            lr=self.args.lr,
                            betas=betas, weight_decay=self.args.weight_decay)
        self.optim_B = Adam(list(self.generator_B.parameters()) + list(self.discriminator_B.parameters()),
                            lr=self.args.lr,
                            betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.generator_A.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

        self.m = nn.Softmax(dim=1)
        self.loss_fct = nn.CrossEntropyLoss()

    def __refresh_training_dataset(self, item_embeddings):
        """
        use for updating item embedding
        """
        user_seq, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
        # training data for node classification
        train_dataset = RecWithContrastiveLearningDataset(self.args, user_seq,
                                                          data_type='train', similarity_model_type='hybrid')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader

    def train(self, epoch):
        # start to use online item similarity
        if epoch > self.args.augmentation_warm_up_epoches:
            print("refresh dataset with updated item embedding")
            self.train_dataloader = self.__refresh_training_dataset(self.generator_A.item_embeddings)
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR, RECALL_1, PRECISION_1 = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR, RECALL_5, PRECISION_5 = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR, RECALL_10, PRECISION_10 = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
            "RECALL@1": '{:.4f}'.format(RECALL_1), "PRECISION@1": '{:.4f}'.format(PRECISION_1),
            "RECALL@5": '{:.4f}'.format(RECALL_5), "PRECISION@5": '{:.4f}'.format(PRECISION_5),
            "RECALL@10": '{:.4f}'.format(RECALL_10), "PRECISION@10": '{:.4f}'.format(PRECISION_10),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR, RECALL_1, PRECISION_1, RECALL_5, PRECISION_5, RECALL_10, PRECISION_10], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, precision, mrr = [], [], [], 0
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
            precision.append(precision_at_k(answers,pred_list,k))

        mrr, mrr_dict = cal_mrr(answers, pred_list)



        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
            "MRR": '{:.4f}'.format(mrr),
            "Recall@5": '{:.4f}'.format(recall[0]), "precision@5": '{:.4f}'.format(precision[0]),
            "Recall@10": '{:.4f}'.format(recall[1]), "precision@10": '{:.4f}'.format(precision[1]),
            "Recall@20": '{:.4f}'.format(recall[3]), "precision@20": '{:.4f}'.format(precision[3]),

        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3], mrr, recall[0],precision[0],recall[1],precision[1],recall[3],precision[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.discriminator_B.cpu().state_dict(), file_name)
        self.discriminator_B.to(self.device)

    def load(self, file_name):
        self.discriminator_B.load_state_dict(torch.load(file_name))

    def _generate_sample(self, probability, pos_ids, neg_ids, neg_nums):
        neg_ids = neg_ids.expand(probability.shape[0], -1)
        # try:
        neg_idxs = torch.multinomial(probability, neg_nums).to(self.device)

        neg_ids = torch.gather(neg_ids, 1, neg_idxs)
        neg_ids = neg_ids.view(-1, self.args.max_seq_length)
        # replace the sampled positive ids with uniform sampled items
        return neg_ids

    def sample_from_generator(self, seq_out, pos_ids, generator):
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * generator.args.max_seq_length).float()  # [batch*seq_len]

        K = int(self.args.item_size * self.args.item_sample_ratio) - 1
        neg_ids = random.sample([i for i in range(1, self.args.item_size)], K)
        neg_ids = torch.tensor(neg_ids, dtype=torch.long).to(self.device)
        neg_emb = generator.item_embeddings(neg_ids)
        full_probability = torch.matmul(seq_emb, neg_emb.transpose(0, 1))
        full_probability = self.m(full_probability) ** self.args.prob_power
        sampled_neg_ids = self._generate_sample(full_probability, pos_ids, neg_ids, 1)

        # replace certain percentage of items as absolute positive items
        replace_idx = (torch.rand(size=(pos_ids.size(0), pos_ids.size(1))) < (1 - self.args.sample_ratio))
        sampled_neg_ids[replace_idx] = pos_ids[replace_idx]
        mask_idx = torch.logical_not(replace_idx).float()
        pos_idx = (pos_ids == sampled_neg_ids).view(pos_ids.size(0) * generator.args.max_seq_length).float()
        neg_idx = (pos_ids != sampled_neg_ids).view(pos_ids.size(0) * generator.args.max_seq_length).float()
        return sampled_neg_ids, pos_idx, neg_idx, mask_idx, istarget


    def discriminator_cross_entropy(self, seq_out, pos_idx, neg_idx, mask_idx, istarget, discriminator):
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        # sum over feature dim
        if self.args.project_type == 'sum':
            neg_logits = torch.sum((seq_emb) / self.args.temperature, -1)
        elif self.args.project_type == 'affine':
            neg_logits = torch.squeeze(discriminator.dis_projection(seq_emb))

        prob_score = torch.sigmoid(neg_logits) + 1e-24
        if self.args.dis_opt_versioin == 'mask_only':
            total_pos_loss = torch.log(prob_score) * istarget * pos_idx * mask_idx
            total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx * mask_idx
        else:
            total_pos_loss = torch.log(prob_score) * istarget * pos_idx
            total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx
        if self.args.dis_loss_type in ['bce']:
            loss = torch.sum(
                - total_pos_loss -
                total_neg_loss
            ) / (torch.sum(istarget))
        return loss


    def generator_cross_entropy(self, seq_out, pos_ids, multi_neg_ids, generator):
        # [batch seq_len hidden_size]
        pos_emb = generator.item_embeddings(pos_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum((pos * seq_emb) / self.args.temperature, -1)  # [batch*seq_len]
        istarget = (pos_ids > 0).view(pos_ids.size(0) * generator.args.max_seq_length).float()  # [batch*seq_len]

        # handle multiple negatives
        total_neg_loss = 0.0

        if self.args.gen_loss_type in ['full-softmax']:
            test_item_emb = generator.item_embeddings.weight
            logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, torch.squeeze(pos_ids.view(-1)))
        return loss

    def CL_generator_cross_entropy(self, seq_out_A, seq_out_B, pos_ids, multi_neg_ids, generator_A, generator_B):
        # [batch seq_len hidden_size]
        pos_emb_A = generator_A.item_embeddings(pos_ids)
        pos_emb_B = generator_B.item_embeddings(pos_ids)
        # [batch*seq_len hidden_size]
        pos_A = pos_emb_A.view(-1, pos_emb_A.size(2))
        pos_B = pos_emb_B.view(-1, pos_emb_B.size(2))
        seq_emb_A = seq_out_A.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        seq_emb_B = seq_out_B.view(-1, self.args.hidden_size)
        pos_logits_A = torch.sum((pos_A * seq_emb_A) / self.args.temperature, -1)  # [batch*seq_len]
        pos_logits_B = torch.sum((pos_B * seq_emb_B) / self.args.temperature, -1)
        # istarget = (pos_ids > 0).view(pos_ids.size(0) * generator.args.max_seq_length).float()  # [batch*seq_len]

        # handle multiple negatives
        total_neg_loss = 0.0

        if self.args.gen_loss_type in ['full-softmax']:
            test_item_emb_A = generator_A.item_embeddings.weight
            logits_A = torch.matmul(seq_emb_A, test_item_emb_A.transpose(0, 1))
            test_item_emb_B = generator_B.item_embeddings.weight
            logits_B = torch.matmul(seq_emb_B, test_item_emb_B.transpose(0, 1))
            logits = torch.sigmoid(logits_B - logits_A) + 1e-24
            loss = self.loss_fct(logits, torch.squeeze(pos_ids.view(-1)))
        return loss

    def cycle_cross_entropy(self, seq_out_a, seq_out_b, generator_C,istarget):
        seq_emb_a = generator_C.item_embeddings(seq_out_a)
        seq_emb_b = generator_C.item_embeddings(seq_out_b)
        neg_logits_a = torch.sum((seq_emb_a) / self.args.temperature, -1)
        neg_logits_b = torch.sum((seq_emb_b) / self.args.temperature, -1)


        prob_score = torch.sigmoid(neg_logits_b - neg_logits_a) + 1e-24
        if self.args.dis_opt_versioin == 'mask_only':
            total_pos_loss = torch.log(prob_score)
            total_neg_loss = torch.log(1 - prob_score)
        else:
            total_pos_loss = torch.log(prob_score)
            total_neg_loss = torch.log(1 - prob_score)
        if self.args.dis_loss_type in ['bce']:
            loss = torch.sum(
                - total_pos_loss -
                total_neg_loss
            )/torch.sum(istarget)
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.discriminator_B.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.discriminator_B.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class ECGANRecTrainer(Trainer):


    def __init__(self, generator_A,
                 generator_B,

                 discriminator_A,
                 discriminator_B,

                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(ECGANRecTrainer, self).__init__(
            generator_A,
            generator_B,

            discriminator_A,
            discriminator_B,

            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.generator_A.transformer_encoder(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0],
                                    cl_output_slice[1])
        return cl_loss

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.generator_A.train()
            self.generator_B.train()


            self.discriminator_A.train()
            self.discriminator_B.train()

            gen_avg_loss = 0.0
            dis_avg_loss = 0.0

            cl_avg_loss = 0.0

            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- generator task ---------------#

                sequence_output_A = self.generator_A.transformer_encoder(input_ids)

                sequence_output_B = self.generator_B.transformer_encoder(input_ids)

                sampled_neg_ids_A, pos_idx_A, neg_idx_A, mask_idx_A, istarget_A = self.sample_from_generator(
                    sequence_output_A, target_pos, self.generator_A)

                sampled_neg_ids_B, pos_idx, neg_idx, mask_idx, istarget = self.sample_from_generator(sequence_output_B,
                                                                                                     target_pos,
                                                                                                     self.generator_B)

                gen_loss = self.generator_cross_entropy(sequence_output_A, target_pos, target_neg, self.generator_A) \
                           + self.generator_cross_entropy(sequence_output_B, target_pos, target_neg, self.generator_B)
                cl_loss_gen = self.CL_generator_cross_entropy(sequence_output_A, sequence_output_B, target_pos, target_neg,
                                                          self.generator_A, self.generator_B)

                # ---------- discriminator task ---------------#

                disc_logits_A = self.discriminator_A.transformer_encoder(sampled_neg_ids_A)
                disc_logits_B = self.discriminator_B.transformer_encoder(sampled_neg_ids_B)


                dis_loss = self.discriminator_cross_entropy(disc_logits_A, pos_idx_A, neg_idx_A, mask_idx_A, istarget_A,
                                                            self.discriminator_A) \
                           + self.discriminator_cross_entropy(disc_logits_B, pos_idx, neg_idx, mask_idx, istarget,
                                                              self.discriminator_B) \




                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)


                joint_loss = self.args.gen_loss_weight * gen_loss + self.args.dis_loss_weight * dis_loss + cl_loss_gen * 0.55
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss

                self.optim_A.zero_grad()
                self.optim_B.zero_grad()

                joint_loss.backward()
                self.optim_A.step()
                self.optim_B.step()


                gen_avg_loss = gen_loss.item()
                dis_avg_loss = dis_loss.item()
                cl_avg_loss = cl_loss_gen.item()

                joint_avg_loss += joint_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "gen_avg_loss": '{:.4f}'.format(gen_avg_loss / len(rec_cf_data_iter)),
                "dis_avg_loss": '{:.4f}'.format(dis_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(
                    cl_sum_avg_loss / (len(rec_cf_data_iter) * self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_' + str(i) + '_loss'] = '{:.4f}'.format(
                    cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                 desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                 total=len(dataloader),
                                 bar_format="{l_bar}{r_bar}")
            self.discriminator_B.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.discriminator_B.transformer_encoder(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.discriminator_B.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)

