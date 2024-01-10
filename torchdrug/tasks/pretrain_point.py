import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_min, scatter, scatter_mean

from torchdrug import core, tasks, layers
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R


import numpy as np
import open3d as o3d
import open3d.core as o3c



@R.register("tasks.AttributeMaskingWithProteinCode")
class AttributeMaskingWithProteinCode(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingWithProteinCode, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

    def preprocess(self, train_set, valid_set, test_set):
        data = train_set[0]
        self.view = getattr(data["graph"], "view", "atom")
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        self.model_output_dim = model_output_dim
        if self.view == "atom":
            num_label = constant.NUM_ATOM
        else:
            num_label = constant.NUM_AMINO_ACID
        self.mlp = layers.MLP(model_output_dim * 2, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        if self.view == "atom":
            target = graph.atom_type[node_index]
            input = graph.node_feature.float()
            input[node_index] = 0
        else:
            target = graph.residue_type[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            # Generate masked edge features. Any better implementation?
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        # TODO: Get protein index of masked nodes
        residue2protein_map = torch.repeat_interleave(num_nodes)
        protein_code = scatter_mean(node_feature, residue2protein_map, dim=0)
        protein_code_repeated = torch.repeat_interleave(protein_code, num_samples, dim=0)
        
        node_feature = node_feature[node_index]
        
        # NOTE: Concatenate node-wise representations with protein code
        node_feature_with_protein_code = torch.cat([node_feature, protein_code_repeated], dim=1)
        
        # pred = self.mlp(node_feature)
        pred = self.mlp(node_feature_with_protein_code)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.AttributeMaskingIndexCode")
class AttributeMaskingIndexCode(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingIndexCode, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

    def preprocess(self, train_set, valid_set, test_set):
        data = train_set[0]
        self.view = getattr(data["graph"], "view", "atom")
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        self.model_output_dim = model_output_dim
        if self.view == "atom":
            num_label = constant.NUM_ATOM
        else:
            num_label = constant.NUM_AMINO_ACID
        self.mlp = layers.MLP(model_output_dim + 1, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        if self.view == "atom":
            target = graph.atom_type[node_index]
            input = graph.node_feature.float()
            input[node_index] = 0
        else:
            target = graph.residue_type[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            # Generate masked edge features. Any better implementation?
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        residue2protein_map = torch.repeat_interleave(num_nodes)
        protein_code = scatter_mean(node_feature, residue2protein_map, dim=0)
        protein_code_repeated = torch.repeat_interleave(protein_code, num_samples, dim=0)
        node_feature_with_protein_code = torch.cat([node_index_org.unsqueeze(1), protein_code_repeated], dim=1)
        pred = self.mlp(node_feature_with_protein_code)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.AttributeMaskingPECode")
class AttributeMaskingPECode(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingPECode, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        self.pe_dim = 128

    def computePositionEncoding(self, d, device='cpu', seq_len=100, n=10000):
        P = torch.zeros((seq_len, d)).to(device)
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def preprocess(self, train_set, valid_set, test_set):
        data = train_set[0]
        self.view = getattr(data["graph"], "view", "atom")
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        self.model_output_dim = model_output_dim
        if self.view == "atom":
            num_label = constant.NUM_ATOM
        else:
            num_label = constant.NUM_AMINO_ACID
        self.mlp = layers.MLP(self.pe_dim + model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        if self.view == "atom":
            target = graph.atom_type[node_index]
            input = graph.node_feature.float()
            input[node_index] = 0
        else:
            target = graph.residue_type[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            # Generate masked edge features. Any better implementation?
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        # residue2protein_map = torch.repeat_interleave(num_nodes)
        # protein_code = scatter_mean(node_feature, residue2protein_map, dim=0)
        # protein_code_repeated = torch.repeat_interleave(protein_code, num_samples, dim=0)
        graph_feature_padded = torch.repeat_interleave(output["graph_feature"], num_samples, dim=0)
        
        
        # NOTE: Get positional encoding for indexes
        node_pe = self.computePositionEncoding(self.pe_dim, graph.device)[node_index_org]
        node_feature_with_protein_code = torch.cat([node_pe, graph_feature_padded], dim=1)
        pred = self.mlp(node_feature_with_protein_code)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.AttributeMaskingRandomPoints")
class AttributeMaskingRandomPoints(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingRandomPoints, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        self.point_dim = 3
        self.mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
    
    def point_cloud_distance(self, target, pred, device):
        assert len(target) == len(pred), "Data size of prediction, target is different"
        pcd_target = o3d.t.geometry.PointCloud(o3c.Tensor(target.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
        pcd_pred = o3d.t.geometry.PointCloud(o3c.Tensor(pred.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
        dists = pcd_target.compute_point_cloud_distance(pcd_pred)
        set_distance = torch.tensor(dists, dtype=torch.float32).mean().to(device)
        return set_distance

    def sample_3d_points(self, num_points, std=0.0, device='cpu'):
        mean, std_dev = 0.0, 0.0
        random_points = torch.normal(mean=mean, std=std_dev, size=(num_points, 3)).to(device)
        return random_points
    
    def sample_mesh_sphere(self, number_of_points, device='cpu'):
        # NOTE: Sample points on a unit mesh(trianlge) sphere
        sampled_points = self.mesh_sphere.sample_points_uniformly(number_of_points)
        sampled_points = torch.tensor(np.asarray(sampled_points.points)).to(device).to(torch.float32)
        
        return sampled_points
    
    def preprocess(self, train_set, valid_set, test_set):
        data = train_set[0]
        self.view = getattr(data["graph"], "view", "atom")
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        self.model_output_dim = model_output_dim
        if self.view == "atom":
            num_label = constant.NUM_ATOM
        else:
            num_label = constant.NUM_AMINO_ACID
        self.mlp = layers.MLP(self.point_dim + model_output_dim, [self.point_dim])
        # self.mlp_z2point = layers.MLP(self.mlp.layers[self.num_mlp_layer-1].out_features, [self.point_dim])
        
    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        if self.view == "residue":
            # target = graph.residue_type[node_index]
            # NOTE: Set targget as 3d coordinates [num nodes * 3]
            target = torch.split(graph.node_position, num_nodes.tolist())
            # Generate masked edge features. Any better implementation?
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Implementation for only residue view"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code by graph representation
        graph_feature_padded = torch.repeat_interleave(output["graph_feature"], num_nodes, dim=0)
        
        # NOTE: Sample random points in 3D / hypersphere
        std = torch.mean(torch.std(graph.node_position, dim=0))
        random_points = self.sample_3d_points(num_cum_nodes[-1], std, graph.device)
        random_points2 = self.sample_mesh_sphere(num_cum_nodes[-1], graph.device)
        point_feature_by_protein_code = torch.cat([random_points, graph_feature_padded], dim=1)
        point_feature_by_protein_code2 = torch.cat([random_points2, graph_feature_padded], dim=1)
        # pred = self.mlp(point_feature_by_protein_code)
        pred = self.mlp(point_feature_by_protein_code2)
        pred = torch.split(pred, num_nodes.tolist())

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        
        # accuracy = (pred.argmax(dim=-1) == target).float().mean()()
        # NOTE: Point cloud distance between two point cloud
        assert len(pred) == len(target), "Number of data in prediction, target is different"
        device = str(pred[0].device)
        batch_set_distance = list(map(lambda x, y: self.point_cloud_distance(x, y, device), pred, target))
        batch_set_distance = torch.tensor(batch_set_distance, dtype=torch.float32, requires_grad=True).mean()
        
        name = tasks._get_metric_name("pcd")
        metric[name] = batch_set_distance

        return metric, batch_set_distance

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric_info, loss = self.evaluate(pred, target)
        metric.update(metric_info)

        # loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("pcd")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.PlddtPredictionWithProteinCode")
class PlddtPredictionWithProteinCode(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(PlddtPredictionWithProteinCode, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

    def preprocess(self, train_set, valid_set, test_set):
        data = train_set[0]
        self.view = getattr(data["graph"], "view", "atom")
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        if self.view == "atom":
            num_label = constant.NUM_ATOM
        else:
            num_label = constant.NUM_AMINO_ACID
        self.mlp = layers.MLP(model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        # NOTE: Taget set by using confience score (plddt) instead of node features
        if self.view == "atom":
            target = graph.atom_type[node_index]
            input = graph.node_feature.float()
            input[node_index] = 0
        else:
            assert graph.residue_type.shape[0] == graph.b_factor.shape[0]
            assert max(node_index) < graph.residue_type.shape[0]
            target = graph.b_factor[node_index]/5
            target = target.long()
            # assert torch.all(target>0)
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            # Generate masked edge features. Any better implementation?
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input = graph.residue_feature.float()

        output = self.model(graph, input, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
        node_feature = node_feature[node_index]
        pred = self.mlp(node_feature)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric



