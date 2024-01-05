import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, datasets, metrics
from torchdrug.core import Registry as R

@R.register("tasks.LinkPrediction")
class LinkPrediction(tasks.Task, core.Configurable):

    def __init__(self, model):
        super(LinkPrediction, self).__init__()
        self.model = model

    def preprocess(self, train_set, valid_set, test_set):
        dataset = train_set.dataset
        graph = dataset.graph
        train_graph = dataset.graph.edge_mask(train_set.indices)

        # flip the edges to make the graph undirected
        edge_list = train_graph.edge_list.repeat(2, 1)
        edge_list[train_graph.num_edge:, :2] = edge_list[train_graph.num_edge:, :2] \
                                               .flip(1)
        index = torch.arange(train_graph.num_edge, device=self.device) \
                .repeat(2, 1).t().flatten()
        data_dict, meta_dict = train_graph.data_mask(edge_index=index)
        train_graph = type(train_graph)(
            edge_list, edge_weight=train_graph.edge_weight[index],
            num_node=train_graph.num_node, num_edge=train_graph.num_edge * 2,
            meta_dict=meta_dict, **data_dict
        )

        self.register_buffer("train_graph", train_graph)
        self.num_node = dataset.num_node

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        metric.update(self.evaluate(pred, target))

        loss = F.binary_cross_entropy_with_logits(pred, target)
        metric["bce loss"] = loss

        all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        neg_batch = torch.randint(self.num_node, batch.shape, device=self.device)
        batch = torch.cat([batch, neg_batch])
        node_in, node_out = batch.t()

        output = self.model(self.train_graph, self.train_graph.node_feature.float(),
                            all_loss, metric)
        node_feature = output["node_feature"]
        pred = torch.einsum("bd, bd -> b",
                            node_feature[node_in], node_feature[node_out])
        return pred

    def target(self, batch):
        batch_size = len(batch)
        target = torch.zeros(batch_size * 2, device=self.device)
        target[:batch_size] = 1
        return target

    def evaluate(self, pred, target):
        roc = metrics.area_under_roc(pred, target)
        return {
            "AUROC": roc
        }