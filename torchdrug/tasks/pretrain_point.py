import copy
import math
import random

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


@R.register("tasks.AttributeMaskingByPoints")
class AttributeMaskingByPoints(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingByPoints, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 512
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512
        self.point_encoder = layers.MLP(self.point_dim, [self.latent_point_dim])

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
            
        self.mapper = layers.MLP(self.latent_point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [num_label])
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        if self.view == "residue":
            target = graph.residue_type[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in AttributeMaskingByPoints"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], torch.as_tensor(num_samples, device=graph.device), dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate sampled coordinates and protein code
        sampled_point_coordinates = self.point_encoder(graph.node_position[node_index])
        points_with_protein_code = torch.cat([sampled_point_coordinates, graph_protein_code], dim=1)
        pred = self.mapper(points_with_protein_code)

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



@R.register("tasks.AttributeMaskingByPointsSet")
class AttributeMaskingByPointsSet(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingByPointsSet, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512

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
            
        self.mapper = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.latent_point_dim] + [num_label])
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        if self.view == "residue":
            target = graph.residue_type[node_index]
            target = torch.split(target, num_samples.tolist())
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in AttributeMaskingByPoints"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], torch.as_tensor(num_samples, device=graph.device), dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate sampled coordinates and protein code
        sampled_point_coordinates = graph.node_position[node_index]
        latent_points = torch.cat([sampled_point_coordinates, graph_protein_code], dim=1)
        pred = self.mapper(latent_points)
        pred = torch.split(pred, num_samples.tolist())

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        # TODO: calculate metric between two discrete distribution
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



@R.register("tasks.AttributeMaskingByPointsFiltered")
class AttributeMaskingByPointsFiltered(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMaskingByPointsFiltered, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512
        self.expand_dim = 512

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
            
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])
        self.expand_coordinates = layers.MLP(3, [self.expand_dim])
        self.mapper = layers.MLP(self.expand_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.latent_point_dim] + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        # NOTE: sample nodes to target ( high b factor or random)
        if self.sample == "high_b_factor":
            node_index_split = torch.split(node_index, num_samples.tolist())
            node_index_split_filter = list(map(lambda x: torch.reshape((graph.b_factor[x] > self.b_factor_threshold).nonzero(), (-1, )), node_index_split))
            num_samples_filtered = list(map(lambda x: x.shape[0], node_index_split_filter))
            node_index_filtered = (graph.b_factor[node_index] > self.b_factor_threshold).nonzero().squeeze()
            node_index = node_index_filtered
            num_samples = num_samples_filtered
            num_sample = sum(num_samples_filtered)

        if self.view == "residue":
            target = graph.residue_type[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in AttributeMaskingByPoints"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], torch.as_tensor(num_samples, device=graph.device), dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate sampled coordinates and protein code
        sampled_point_coordinates = graph.node_position[node_index]
        expanded_point = self.expand_coordinates(sampled_point_coordinates)
        latent_points = torch.cat([expanded_point, graph_protein_code], dim=1)
        pred = self.mapper(latent_points)

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


@R.register("tasks.ConfidenceScoreByPoints")
class ConfidenceScoreByPoints(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(ConfidenceScoreByPoints, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.b_factor_filter = False
        self.b_factor_threshold = 80.0
        self.reduced_graph_dimension = 512

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
            
        self.mapper = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.latent_point_dim] + [10])
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        # NOTE: sample nodes to target
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        if self.view == "residue":
            target = (graph.b_factor[node_index] / 10).to(torch.int64)
            # TODO: given [1, 2, 1], reshape it to 
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in AttributeMaskingByPoints"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], num_samples, dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate sampled coordinates and protein code
        latent_points = torch.cat([graph.node_position[node_index], graph_protein_code], dim=1)
        pred = self.mapper(latent_points)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        # mae = F.l1_loss(pred, target, reduction="mean")
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


@R.register("tasks.DenoisingStructure")
class DenoisingStructure(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(DenoisingStructure, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512

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
            
        self.mapper = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.point_dim] )
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        if self.view == "residue":
            target = graph.node_position[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in AttributeMaskingByPoints"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], num_samples, dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate sampled coordinates and protein code
        noisy_coordinates = graph.node_position[node_index] + torch.randn_like(graph.node_position[node_index]) * 0.1
        latent_points = torch.cat([noisy_coordinates, graph_protein_code], dim=1)
        pred = self.mapper(latent_points)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}

        l1_loss = F.l1_loss(pred, target)
        name = "mae"
        metric[name] = l1_loss

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.l1_loss(pred, target)
        name = tasks._get_criterion_name("mae")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.ProteinWorkshopDenoising")
class ProteinWorkshopDenoising(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(ProteinWorkshopDenoising, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512
        self.noise_strategy = "uniform"
        # self.noise_strategy = "gaussian"

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
            
        self.mapper = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.point_dim] )
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        # NOTE: Add noise to graph positions
        org_node_position = graph.node_position
        noise = torch.randn_like(graph.node_position) * 0.1 if self.noise_strategy == "gaussian" else (torch.rand_like(graph.node_position) - 0.5) * 2 * 0.1
        graph.node_position += noise
        
        if self.view == "residue":
            target = org_node_position[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in AttributeMaskingByPoints"

        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        pred = self.mapper(latent_points)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}

        l1_loss = F.l1_loss(pred, target)
        name = "mae"
        metric[name] = l1_loss

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.l1_loss(pred, target)
        name = tasks._get_criterion_name("mae")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric




@R.register("tasks.RandomNoiseMatching")
class RandomNoiseMatching(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(RandomNoiseMatching, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512

    def set_distance(self, x, y):
        # NOTE: Find the nearest point in y for each point in x, calculate the mean distance
        distance_matrix = torch.cdist(x, y, p=2)
        distance = distance_matrix.min(dim=1)[0].mean()
        return distance

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
            
        self.mapper = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.latent_point_dim] + [self.point_dim])
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        # NOTE: Sample nodes to target
        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        if self.view == "residue":
            target = graph.node_position[node_index]
            target = torch.split(target, num_samples.tolist())
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in RandomNoiseMatching"
        
        # NOTE: Generate node representations
        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], torch.as_tensor(num_samples, device=graph.device), dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate random coordinates and protein code
        random_coordinates = torch.randn_like(graph.node_position[node_index])
        latent_points = torch.cat([random_coordinates, graph_protein_code], dim=1)
        pred = self.mapper(latent_points)
        pred = torch.split(pred, num_samples.tolist())

        return pred, target
    
    def evaluate(self, pred, target):
        metric = {}

        # l1_loss = F.l1_loss(pred, target)
        set_distance_list_x_to_y = list(map(lambda x, y: self.set_distance(x, y).unsqueeze(dim=0), pred, target))
        set_distance_list_y_to_x = list(map(lambda x, y: self.set_distance(y, x).unsqueeze(dim=0), pred, target))
        set_distance = torch.cat(set_distance_list_x_to_y).mean() + torch.cat(set_distance_list_y_to_x).mean()
        
        name = "sd"
        metric[name] = set_distance

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        # loss = F.l1_loss(pred, target)
        name = tasks._get_criterion_name("mae")
        metric[name] = metric["sd"]

        all_loss += metric["sd"]

        return all_loss, metric


@R.register("tasks.UniformNoiseMatching")
class UniformNoiseMatching(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(UniformNoiseMatching, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        
        self.point_dim = 3
        self.latent_point_dim = 128
        
        self.sample = "high_b_factor"
        self.b_factor_filter = False
        self.b_factor_threshold = 90.0
        self.reduced_graph_dimension = 512

    def set_distance(self, x, y):
        # NOTE: Find the nearest point in y for each point in x, calculate the mean distance
        distance_matrix = torch.cdist(x, y, p=2)
        distance = distance_matrix.min(dim=1)[0].mean()
        return distance

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
            
        self.mapper = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.reduced_graph_dimension] * (self.num_mlp_layer - 1) + [self.latent_point_dim] + [self.point_dim])
        self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        # NOTE: Sample nodes to target
        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        self.num_nodes = num_nodes.tolist()
        self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index_org = node_index
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        
        if self.view == "residue":
            target = graph.node_position[node_index]
            target = torch.split(target, num_samples.tolist())
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input_graph = graph.residue_feature.float()
        else:
            assert "Do not use atom view in RandomNoiseMatching"
        
        # NOTE: Generate node representations
        output = self.model(graph, input_graph, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
            
        # NOTE: Make protein code
        graph_protein_code = torch.repeat_interleave(output["graph_feature"], torch.as_tensor(num_samples, device=graph.device), dim=0)
        graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        # NOTE: Concatenate random coordinates and protein code
        random_coordinates = (torch.rand_like(graph.node_position[node_index]) - 0.5) * 2
        latent_points = torch.cat([random_coordinates, graph_protein_code], dim=1)
        pred = self.mapper(latent_points)
        pred = torch.split(pred, num_samples.tolist())

        return pred, target
    
    def evaluate(self, pred, target):
        metric = {}

        # l1_loss = F.l1_loss(pred, target)
        set_distance_list_x_to_y = list(map(lambda x, y: self.set_distance(x, y).unsqueeze(dim=0), pred, target))
        set_distance_list_y_to_x = list(map(lambda x, y: self.set_distance(y, x).unsqueeze(dim=0), pred, target))
        set_distance = torch.cat(set_distance_list_x_to_y).mean() + torch.cat(set_distance_list_y_to_x).mean()
        
        name = "sd"
        metric[name] = set_distance

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        # loss = F.l1_loss(pred, target)
        name = tasks._get_criterion_name("mae")
        metric[name] = metric["sd"]

        all_loss += metric["sd"]

        return all_loss, metric