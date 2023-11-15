import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RunTsne():
    def __init__(self,
                selected_cls,        # 选择可视化几个类别
                domId2name,          # 不同域的ID
                trainId2name,        # 标签中每个ID所对应的类别
                trainId2color=None,  # 标签中每个ID所对应的颜色
                output_dir='./',     # 保存的路径
                tsnecuda=True,       # 是否使用tsnecuda，如果不使用tsnecuda就使用MulticoreTSNE
                extention='.png',    # 保存图片的格式
                duplication=10):     # 程序循环运行几次，即保存多少张结果图片
        self.tsne_path = output_dir
        os.makedirs(self.tsne_path, exist_ok=True)
        self.domId2name = domId2name
        self.name2domId = {v:k for k,v in domId2name.items()}
        self.trainId2name = trainId2name
        self.trainId2color = trainId2color
        self.selected_cls = selected_cls
        self.name2trainId = {v:k for k,v in trainId2name.items()}
        self.selected_clsid = [self.name2trainId[x] for x in selected_cls]
        self.tsnecuda = tsnecuda
        self.extention = extention
        self.num_class = 19
        self.duplication = duplication

        self.init_basket()    # 初始化

        if self.tsnecuda:
            from tsnecuda import TSNE
            self.max_pointnum = 9000    # 最大特征向量的数量
            self.perplexity = 30        
            self.learning_rate = 100    # t-SNE的学习率
            self.n_iter = 4000          # t-SNE迭代步数
            self.num_neighbors = 128    
            self.TSNE = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate, metric='innerproduct',
                 random_seed=2, num_neighbors=self.num_neighbors, n_iter=self.n_iter, verbose=1)
        else:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            self.max_pointnum = 10200
            self.perplexity = 50
            self.learning_rate = 4800
            self.n_iter = 3000
            self.TSNE = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                             n_iter=self.n_iter, verbose=1, n_jobs=4)

    def init_basket(self):
        self.feat_vecs = torch.tensor([]).cuda()            # 特征向量
        self.feat_vec_labels = torch.tensor([]).cuda()      # 特征向量的类别
        self.feat_vec_domlabels = torch.tensor([]).cuda()   # 特征向量的域信息

    def input2basket(self, feature_map, gt_cuda, datasetname):
        b, c, h, w = feature_map.shape
        features = F.normalize(feature_map.clone(), dim=1)
        gt_cuda = gt_cuda.clone()
        H, W = gt_cuda.size()[-2:]
        gt_cuda[gt_cuda == 255] = self.num_class
        gt_cuda = F.one_hot(gt_cuda, num_classes=self.num_class + 1)

        gt = gt_cuda.view(1, -1, self.num_class + 1)
        denominator = gt.sum(1).unsqueeze(dim=1)
        denominator = denominator.sum(0)  # batchwise sum
        denominator = denominator.squeeze()

        features = F.interpolate(features, [H, W], mode='bilinear', align_corners=True)
        # 这里是将feature采样到跟标签一样的大小。当然也可以将标签采样到跟feature一样的大小
        features = features.view(b, c, -1)
        nominator = torch.matmul(features, gt.type(torch.float32))
        nominator = torch.t(nominator.sum(0))  # batchwise sum

        for slot in self.selected_clsid:
            if denominator[slot] != 0:
                cls_vec = nominator[slot] / denominator[slot]  # mean vector
                cls_label = (torch.zeros(1, 1) + slot).cuda()
                dom_label = (torch.zeros(1, 1) + self.name2domId[datasetname]).cuda()
                self.feat_vecs = torch.cat((self.feat_vecs, cls_vec.unsqueeze(dim=0)), dim=0)
                self.feat_vec_labels = torch.cat((self.feat_vec_labels, cls_label), dim=0)
                self.feat_vec_domlabels = torch.cat((self.feat_vec_domlabels, dom_label), dim=0)

    def draw_tsne(self, domains2draw, adding_name=None, plot_memory=False, clscolor=True):
        feat_vecs_temp = F.normalize(self.feat_vecs.clone(), dim=1).cpu().numpy()
        feat_vec_labels_temp = self.feat_vec_labels.clone().to(torch.int64).squeeze().cpu().numpy()
        feat_vec_domlabels_temp = self.feat_vec_domlabels.clone().to(torch.int64).squeeze().cpu().numpy()

        # if adding_name is not None:
        #     tsne_file_name = adding_name +'_tSNE_' + ''.join(domains2draw) + '_' + str(self.perplexity) + '_' + str(self.learning_rate)
        # else:
        #     tsne_file_name = 'tSNE_' + ''.join(domains2draw) + '_' + str(self.perplexity) + '_' + str(self.learning_rate)

        if clscolor:
            sequence_of_colors = np.array([list(self.trainId2color[x]) for x in range(19)])/255.0
        else:
            sequence_of_colors = ["tab:purple", "tab:pink", "lightgray","dimgray","yellow","tab:brown","tab:orange","blue","tab:green","darkslategray","tab:cyan","tab:red","lime","tab:blue","navy","tab:olive","blueviolet", "deeppink","red"]
            sequence_of_colors[1] = "tab:olive"
            sequence_of_colors[2] = "tab:grey"
            sequence_of_colors[5] = "tab:cyan"
            sequence_of_colors[8] =  "tab:pink"
            sequence_of_colors[10] = "tab:brown"
            sequence_of_colors[13] = "tab:red"

        name2domId = {self.domId2name[x] : x for x in self.domId2name.keys()}
        domIds2draw = [name2domId[x] for x in domains2draw]
        name2trainId = {v:k for k,v in self.trainId2name.items()}
        trainIds2draw = [name2trainId[x] for x in self.selected_cls]
        domain_color = ["tab:blue", "tab:green","tab:orange","tab:purple","black"]
        assert len(feat_vec_domlabels_temp.shape) == 1
        assert len(feat_vecs_temp.shape) == 2
        assert len(feat_vec_labels_temp.shape) == 1

        # domain spliting
        dom_idx = np.array([x in domIds2draw for x in feat_vec_domlabels_temp])
        feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = feat_vecs_temp[dom_idx, :], feat_vec_labels_temp[dom_idx], \
                                                                       feat_vec_domlabels_temp[dom_idx]

        # max_pointnum random sampling.
        if feat_vecs_temp.shape[0] > self.max_pointnum:
            pointnum_predraw = feat_vec_labels_temp.shape[0]
            dom_idx = np.random.randint(0,pointnum_predraw,self.max_pointnum)
            feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = feat_vecs_temp[dom_idx, :], feat_vec_labels_temp[dom_idx], feat_vec_domlabels_temp[dom_idx]

        vecs2tsne = feat_vecs_temp
        
        perplexity_list = [50]
        learning_list = [4800]
        iter_list = [3000]
        for _, per in enumerate(perplexity_list):
            for i, lr in enumerate(learning_list):
                if adding_name is not None:
                    tsne_file_name = str(per) + '_' + str(lr) +'_'+ adding_name +'_tSNE_' + ''.join(domains2draw)
                else:
                    tsne_file_name = str(per) + '_' + str(lr) +'_'+ 'tSNE_' + ''.join(domains2draw)
                from MulticoreTSNE import MulticoreTSNE as TSNE
                self.TSNE = TSNE(n_components=2, perplexity=per, learning_rate=lr,
                             n_iter=iter_list[i], verbose=1, n_jobs=4)
                for tries in range(self.duplication):
                    X_embedded = self.TSNE.fit_transform(vecs2tsne)
                    print('\ntsne done')
                    X_embedded[:,0] = (X_embedded[:,0] - X_embedded[:,0].min()) / (X_embedded[:,0].max() - X_embedded[:,0].min())
                    X_embedded[:,1] = (X_embedded[:,1] - X_embedded[:,1].min()) / (X_embedded[:,1].max() - X_embedded[:,1].min())

                    feat_coords = X_embedded

                    ##### color means class
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111)
                    dom_marker = ['x','+','P','v','s']
                    for i ,dom_i in enumerate(domIds2draw):
                        for cls_i in trainIds2draw:
                            temp_coords = feat_coords[(feat_vec_labels_temp == cls_i) & (feat_vec_domlabels_temp == dom_i),:]
                            ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                                    color=sequence_of_colors[cls_i], label=self.domId2name[dom_i]+'_'+self.trainId2name[cls_i], s=50, marker = dom_marker[i])

                    print('scatter plot done')
                    lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    tsne_file_name_c = str(tries)+'_ColorClass_'+tsne_file_name+self.extention
                    tsne_file_path = os.path.join(self.tsne_path,tsne_file_name_c)
                    fig.savefig(tsne_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
                    # plt.show()
                    fig.clf()

                    ##### color means domains
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111)

                    for i ,dom_i in enumerate(domIds2draw):
                        for cls_i in trainIds2draw:
                            temp_coords = feat_coords[(feat_vec_labels_temp == cls_i) & (feat_vec_domlabels_temp == dom_i),:]
                            ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                                    color= domain_color[dom_i], label=self.domId2name[dom_i]+'_'+self.trainId2name[cls_i], s=50, marker = dom_marker[i])

                    print('scatter plot done')
                    lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    tsne_file_name_d = str(tries) + '_ColorDomain_' + tsne_file_name + self.extention
                    tsne_file_path = os.path.join(self.tsne_path,tsne_file_name_d)
                    fig.savefig(tsne_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
                    # plt.show()
                    fig.clf()

        return tsne_file_path
