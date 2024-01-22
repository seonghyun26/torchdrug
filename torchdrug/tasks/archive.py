# @R.register("tasks.AttributeMaskingIndexCode")
# class AttributeMaskingIndexCode(tasks.Task, core.Configurable):
#     """
#     Parameters:
#         model (nn.Module): node representation model
#         mask_rate (float, optional): rate of masked nodes
#         num_mlp_layer (int, optional): number of MLP layers
#     """

#     def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
#         super(AttributeMaskingIndexCode, self).__init__()
#         self.model = model
#         self.mask_rate = mask_rate
#         self.num_mlp_layer = num_mlp_layer
#         self.graph_construction_model = graph_construction_model

#     def preprocess(self, train_set, valid_set, test_set):
#         data = train_set[0]
#         self.view = getattr(data["graph"], "view", "atom")
#         if hasattr(self.model, "node_output_dim"):
#             model_output_dim = self.model.node_output_dim
#         else:
#             model_output_dim = self.model.output_dim
#         self.model_output_dim = model_output_dim
#         if self.view == "atom":
#             num_label = constant.NUM_ATOM
#         else:
#             num_label = constant.NUM_AMINO_ACID
#         self.mlp = layers.MLP(model_output_dim + 1, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

#     def predict_and_target(self, batch, all_loss=None, metric=None):
#         graph = batch["graph"]
#         if self.graph_construction_model:
#             graph = self.graph_construction_model.apply_node_layer(graph)

#         num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
#         num_cum_nodes = num_nodes.cumsum(0)
#         num_samples = (num_nodes * self.mask_rate).long().clamp(1)
#         num_sample = num_samples.sum()
#         sample2graph = torch.repeat_interleave(num_samples)
#         node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
#         node_index_org = node_index
#         node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

#         if self.view == "atom":
#             target = graph.atom_type[node_index]
#             input = graph.node_feature.float()
#             input[node_index] = 0
#         else:
#             target = graph.residue_type[node_index]
#             with graph.residue():
#                 graph.residue_feature[node_index] = 0
#                 graph.residue_type[node_index] = 0
#             # Generate masked edge features. Any better implementation?
#             if self.graph_construction_model:
#                 graph = self.graph_construction_model.apply_edge_layer(graph)
#             input_graph = graph.residue_feature.float()

#         output = self.model(graph, input_graph, all_loss, metric)
#         if self.view in ["node", "atom"]:
#             node_feature = output["node_feature"]
#         else:
#             node_feature = output.get("residue_feature", output.get("node_feature"))
            
#         # NOTE: Make protein code
#         residue2protein_map = torch.repeat_interleave(num_nodes)
#         protein_code = scatter_mean(node_feature, residue2protein_map, dim=0)
#         protein_code_repeated = torch.repeat_interleave(protein_code, num_samples, dim=0)
#         node_feature_with_protein_code = torch.cat([node_index_org.unsqueeze(1), protein_code_repeated], dim=1)
#         pred = self.mlp(node_feature_with_protein_code)

#         return pred, target

#     def evaluate(self, pred, target):
#         metric = {}
#         accuracy = (pred.argmax(dim=-1) == target).float().mean()

#         name = tasks._get_metric_name("acc")
#         metric[name] = accuracy

#         return metric

#     def forward(self, batch):
#         """"""
#         all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
#         metric = {}

#         pred, target = self.predict_and_target(batch, all_loss, metric)
#         metric.update(self.evaluate(pred, target))

#         loss = F.cross_entropy(pred, target)
#         name = tasks._get_criterion_name("ce")
#         metric[name] = loss

#         all_loss += loss

#         return all_loss, metric


# @R.register("tasks.AttributeMaskingPECode")
# class AttributeMaskingPECode(tasks.Task, core.Configurable):
#     """
#     Parameters:
#         model (nn.Module): node representation model
#         mask_rate (float, optional): rate of masked nodes
#         num_mlp_layer (int, optional): number of MLP layers
#     """

#     def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
#         super(AttributeMaskingPECode, self).__init__()
#         self.model = model
#         self.mask_rate = mask_rate
#         self.num_mlp_layer = num_mlp_layer
#         self.graph_construction_model = graph_construction_model
#         self.pe_dim = 128

#     def computePositionEncoding(self, d, device='cpu', seq_len=100, n=10000):
#         P = torch.zeros((seq_len, d)).to(device)
#         for k in range(seq_len):
#             for i in np.arange(int(d/2)):
#                 denominator = np.power(n, 2*i/d)
#                 P[k, 2*i] = np.sin(k/denominator)
#                 P[k, 2*i+1] = np.cos(k/denominator)
#         return P

#     def preprocess(self, train_set, valid_set, test_set):
#         data = train_set[0]
#         self.view = getattr(data["graph"], "view", "atom")
#         if hasattr(self.model, "node_output_dim"):
#             model_output_dim = self.model.node_output_dim
#         else:
#             model_output_dim = self.model.output_dim
#         self.model_output_dim = model_output_dim
#         if self.view == "atom":
#             num_label = constant.NUM_ATOM
#         else:
#             num_label = constant.NUM_AMINO_ACID
#         self.mlp = layers.MLP(self.pe_dim + model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

#     def predict_and_target(self, batch, all_loss=None, metric=None):
#         graph = batch["graph"]
#         if self.graph_construction_model:
#             graph = self.graph_construction_model.apply_node_layer(graph)

#         num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
#         num_cum_nodes = num_nodes.cumsum(0)
#         num_samples = (num_nodes * self.mask_rate).long().clamp(1)
#         num_sample = num_samples.sum()
#         sample2graph = torch.repeat_interleave(num_samples)
#         node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
#         node_index_org = node_index
#         node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

#         if self.view == "atom":
#             target = graph.atom_type[node_index]
#             input = graph.node_feature.float()
#             input[node_index] = 0
#         else:
#             target = graph.residue_type[node_index]
#             with graph.residue():
#                 graph.residue_feature[node_index] = 0
#                 graph.residue_type[node_index] = 0
#             # Generate masked edge features. Any better implementation?
#             if self.graph_construction_model:
#                 graph = self.graph_construction_model.apply_edge_layer(graph)
#             input_graph = graph.residue_feature.float()

#         output = self.model(graph, input_graph, all_loss, metric)
#         if self.view in ["node", "atom"]:
#             node_feature = output["node_feature"]
#         else:
#             node_feature = output.get("residue_feature", output.get("node_feature"))
            
#         # NOTE: Make protein code
#         # residue2protein_map = torch.repeat_interleave(num_nodes)
#         # protein_code = scatter_mean(node_feature, residue2protein_map, dim=0)
#         # protein_code_repeated = torch.repeat_interleave(protein_code, num_samples, dim=0)
#         graph_feature_padded = torch.repeat_interleave(output["graph_feature"], num_samples, dim=0)
        
        
#         # NOTE: Get positional encoding for indexes
#         node_pe = self.computePositionEncoding(self.pe_dim, graph.device)[node_index_org]
#         node_feature_with_protein_code = torch.cat([node_pe, graph_feature_padded], dim=1)
#         pred = self.mlp(node_feature_with_protein_code)

#         return pred, target

#     def evaluate(self, pred, target):
#         metric = {}
#         accuracy = (pred.argmax(dim=-1) == target).float().mean()

#         name = tasks._get_metric_name("acc")
#         metric[name] = accuracy

#         return metric

#     def forward(self, batch):
#         """"""
#         all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
#         metric = {}

#         pred, target = self.predict_and_target(batch, all_loss, metric)
#         metric.update(self.evaluate(pred, target))

#         loss = F.cross_entropy(pred, target)
#         name = tasks._get_criterion_name("ce")
#         metric[name] = loss

#         all_loss += loss

#         return all_loss, metric


# @R.register("tasks.GaussianSpacePointMatching")
# class GaussianSpacePointMatching(tasks.Task, core.Configurable):
#     """
#     Parameters:
#         model (nn.Module): node representation model
#         mask_rate (float, optional): rate of masked nodes
#         num_mlp_layer (int, optional): number of MLP layers
#     """

#     def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
#         super(GaussianSpacePointMatching, self).__init__()
#         self.model = model
#         self.mask_rate = mask_rate
#         self.num_mlp_layer = num_mlp_layer
#         self.graph_construction_model = graph_construction_model
        
#         self.point_dim = 3
#         self.sample_3d_rate = 5
        
#         self.b_factor_filter = False
#         self.b_factor_threshold = 80.0
#         self.reduced_graph_dimension = 512
#         self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])
        
    
#     def point_cloud_distance(self, target, pred, device):
#         assert len(target) == len(pred), "Data size of prediction, target is different"
#         pcd_target = o3d.t.geometry.PointCloud(o3c.Tensor(target.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
#         pcd_pred = o3d.t.geometry.PointCloud(o3c.Tensor(pred.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
#         dists = pcd_target.compute_point_cloud_distance(pcd_pred)
#         set_distance = torch.tensor(dists, dtype=torch.float32).mean().to(device)
#         return set_distance

#     def sample_3d_points(self, num_points, std=0.0, device='cpu'):
#         mean, std_dev = 0.0, 0.0
#         random_points = torch.normal(mean=mean, std=std_dev, size=(num_points, 3)).to(device)
#         return random_points
    
#     def preprocess(self, train_set, valid_set, test_set):
#         data = train_set[0]
#         self.view = getattr(data["graph"], "view", "atom")
#         if hasattr(self.model, "node_output_dim"):
#             model_output_dim = self.model.node_output_dim
#         else:
#             model_output_dim = self.model.output_dim
#         self.model_output_dim = model_output_dim
#         if self.view == "atom":
#             num_label = constant.NUM_ATOM
#         else:
#             num_label = constant.NUM_AMINO_ACID
#         self.mlp = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.point_dim])
#         self.mlp2 = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.point_dim])
#         self.encoder = self.mlp
#         self.decoder = self.mlp2
        
#     def predict_and_target(self, batch, all_loss=None, metric=None):
#         graph = batch["graph"]
        
#         if self.graph_construction_model:
#             graph = self.graph_construction_model.apply_node_layer(graph)

#         num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
#         num_cum_nodes = num_nodes.cumsum(0)
        
#         # NOTE: sampling nodes
#         # num_samples = (num_nodes * self.mask_rate).long().clamp(1)
#         # num_sample = num_samples.sum()
#         # sample2graph = torch.repeat_interleave(num_samples)
#         # node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
#         # node_index_org = node_index
#         # node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
#         self.num_nodes = num_nodes.tolist()
#         self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)

#         if self.view == "residue":
#             # target = graph.residue_type[node_index]
#             # NOTE: Set u taret as 3d coordinates [num nodes * 3]
#             target = torch.split(graph.node_position, num_nodes.tolist())
#             # Generate masked edge features. Any better implementation?
#             if self.graph_construction_model:
#                 graph = self.graph_construction_model.apply_edge_layer(graph)
#             input_graph = graph.residue_feature.float()
#         else:
#             assert "Implementation for only residue view"

#         # NOTE: Pretrain model
#         output = self.model(graph, input_graph, all_loss, metric)
#         if self.view in ["node", "atom"]:
#             node_feature = output["node_feature"]
#         else:
#             node_feature = output.get("residue_feature", output.get("node_feature"))
            
#         # NOTE: sample nodes by high plddt scores
#         if self.b_factor_filter:
#             b_factor_node_index = (graph.b_factor > self.b_factor_threshold).nonzero(as_tuple=True)[0]
#             node_feature = node_feature[b_factor_node_index]
        
#         '''
#             NOTE: two point set generated
#             0. Generate protein code
#             1. "point + protein code" embedded to primitive(gaussian) space by self.mlp (encoder part)
#             2. Random points sampled from 3D / hypersphere (decoder part)
#         '''
#         graph_protein_code = torch.repeat_interleave(output["graph_feature"], num_nodes, dim=0)
#         # graph_protein_code = self.graph_representation_reduction(graph_protein_code)
        
#         # z_points = torch.cat([graph.node_position, graph_protein_code], dim=1)
#         # embedded_primitive_space_points = self.encoder(z_points)
        
#         # std = torch.mean(torch.std(graph.node_position, dim=0))
#         # random_gaussian_space_points = self.sample_3d_points(num_cum_nodes[-1] * self.sample_3d_rate, std, graph.device)
#         # random_gaussian_space_points = torch.cat([random_gaussian_space_points, graph_protein_code], dim=1)
#         # random_gaussian_space_points = self.decoder(random_gaussian_space_points)
#         # random_gaussian_space_points = torch.split(random_gaussian_space_points, self.num_nodes)
        
#         return embedded_primitive_space_points, random_gaussian_space_points, target

#     def evaluate(self, u_space_points, s_space_points, s_space_target):
#         metric = {}
        
#         # NOTE: Loss for reconstruction (s space), set distance between two point cloud
#         assert len(s_space_points) == len(s_space_target), "Number of data in prediction, target is different"
#         device = str(s_space_points[0].device)
#         s_space_distance = list(map(lambda x, y: self.point_cloud_distance(x, y, device), s_space_points, s_space_target))
#         s_space_distance = list(map(lambda x, y: x / y, s_space_distance, self.num_nodes))
#         s_space_distance = torch.tensor(s_space_distance, dtype=torch.float32, requires_grad=True).mean()
        
#         # NOTE: Loss for charformer distance (primitive space, u), distance from encoded points to mesh
#         # mesh_points = self.mesh_sphere.sample_points_poisson_disk(number_of_points=self.mesh_number_of_points)
#         gaussian_points = self.sample_3d_points(u_space_points.shape[0] * self.sample_3d_rate, device=device)
#         embedded_points = o3d.t.geometry.PointCloud(o3c.Tensor(u_space_points.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
#         u_space_distance = embedded_points.compute_point_cloud_distance(gaussian_points)
#         u_space_distance = torch.tensor(u_space_distance, dtype=torch.float32, requires_grad=True).mean()
        
#         '''
#         NOTE: mean of mean is different to just mean since the number of points are different
#         But probably won't make much difference
#         '''
#         # primitive_space_distance = torch.split(primitive_space_distance, self.num_nodes)
#         # primitive_space_distance = list(map(lambda x: torch.mean(x), primitive_space_distance))
#         # primitive_space_distance = torch.tensor(primitive_space_distance, dtype=torch.float32, requires_grad=True).mean()
        
#         name_s = tasks._get_metric_name("distance_3d")
#         name_u = tasks._get_metric_name("distance_primitive")
#         metric[name_s] = s_space_distance
#         metric[name_u] = u_space_distance

#         return metric, s_space_distance, u_space_distance

#     def forward(self, batch):
#         """"""
#         all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
#         metric = {}

#         u_space_points, s_space_points, s_space_target = self.predict_and_target(batch, all_loss, metric)
#         metric_info, loss_rec, loss_cd = self.evaluate(u_space_points, s_space_points, s_space_target)
#         metric.update(metric_info)

#         loss = loss_rec + loss_cd
#         name = tasks._get_criterion_name("distance_loss")
#         metric[name] = loss

#         all_loss += loss

#         return all_loss, metric



# @R.register("tasks.PrimitiveSpacePointMatching")
# class PrimitiveSpacePointMatching(tasks.Task, core.Configurable):
#     """
#     Parameters:
#         model (nn.Module): node representation model
#         mask_rate (float, optional): rate of masked nodes
#         num_mlp_layer (int, optional): number of MLP layers
#     """

#     def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
#         super(PrimitiveSpacePointMatching, self).__init__()
#         self.model = model
#         self.mask_rate = mask_rate
#         self.num_mlp_layer = num_mlp_layer
#         self.graph_construction_model = graph_construction_model
        
#         self.point_dim = 3
#         self.mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
#         self.mesh_number_of_points = max(10000, len(self.mesh_sphere.vertices))
        
#         self.b_factor_filter = False
#         self.b_factor_threshold = 80.0
#         self.reduced_graph_dimension = 512
    
#     def normalize_points(self, point_list):
#         point_max, point_min = point_list.max(dim=0)[0], point_list.min(dim=0)[0]
#         normalzied_points = (point_list - point_min) / (point_max - point_min)
#         return normalzied_points
    
#     def point_cloud_distance(self, target, pred, device):
#         assert len(target) == len(pred), "Data size of prediction, target is different"
#         pcd_target = o3d.t.geometry.PointCloud(o3c.Tensor(target.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
#         pcd_pred = o3d.t.geometry.PointCloud(o3c.Tensor(pred.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
#         dists = pcd_target.compute_point_cloud_distance(pcd_pred)
#         set_distance = torch.tensor(dists, dtype=torch.float32).sum().to(device)
#         return set_distance
    
#     def sample_mesh_sphere(self, number_of_points, device='cpu'):
#         # NOTE: Sample points on a unit mesh(trianlge) sphere
#         sampled_points = self.mesh_sphere.sample_points_uniformly(number_of_points)
#         sampled_points = torch.tensor(np.asarray(sampled_points.points)).to(device).to(torch.float32)
        
#         return sampled_points
    
#     def preprocess(self, train_set, valid_set, test_set):
#         data = train_set[0]
#         self.view = getattr(data["graph"], "view", "atom")
#         if hasattr(self.model, "node_output_dim"):
#             model_output_dim = self.model.node_output_dim
#         else:
#             model_output_dim = self.model.output_dim
#         self.model_output_dim = model_output_dim
#         if self.view == "atom":
#             num_label = constant.NUM_ATOM
#         else:
#             num_label = constant.NUM_AMINO_ACID
#         self.mlp = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.point_dim])
#         self.mlp2 = layers.MLP(self.point_dim + self.reduced_graph_dimension, [self.point_dim])
#         self.graph_representation_reduction = layers.MLP(model_output_dim, [self.reduced_graph_dimension])
#         self.mapper = self.mlp2
        
#     def predict_and_target(self, batch, all_loss=None, metric=None):
#         # graph = batch["graph"]
#         # if self.graph_construction_model:
#         #     graph = self.graph_construction_model.apply_node_layer(graph)
#         # NOTE: Construct graph data, compute related attributes
#         graph = self.graph_construction_model.apply_node_layer(batch["graph"]) if self.graph_construction_model else batch["graph"]
#         num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
#         num_cum_nodes = num_nodes.cumsum(0)
#         self.num_nodes = num_nodes.tolist()
#         self.num_cum_nodes = torch.cat((torch.zeros(1).to(graph.device), num_cum_nodes), dim=0).to(torch.int64)
        
#         # NOTE: sampling nodes
#         # num_samples = (num_nodes * self.mask_rate).long().clamp(1)
#         # num_sample = num_samples.sum()
#         # sample2graph = torch.repeat_interleave(num_samples)
#         # node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
#         # node_index_org = node_index
#         # node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

#         # NOTE: Set target tensor as 3d coordinates, shape [num_nodes * 3]
#         if self.view == "residue":
#             target = torch.split(graph.node_position, num_nodes.tolist())
#             # target = list(map(lambda x : x / torch.std(x), target))
#             # Generate masked edge features. Any better implementation?
#             if self.graph_construction_model:
#                 graph = self.graph_construction_model.apply_edge_layer(graph)
#             input_graph = graph.residue_feature.float()
#         else:
#             assert "Implementation for only residue view"

#         # NOTE: Pretrain model
#         output = self.model(graph, input_graph, all_loss, metric)
#         if self.view in ["node", "atom"]:
#             node_feature = output["node_feature"]
#         else:
#             node_feature = output.get("residue_feature", output.get("node_feature"))
            
#         # NOTE: sample nodes by high plddt scores
#         if self.b_factor_filter:
#             b_factor_node_index = (graph.b_factor > self.b_factor_threshold).nonzero(as_tuple=True)[0]
#             node_feature = node_feature[b_factor_node_index]
        
#         '''
#             NOTE: two point set generated
#             1. "point + protein code" embedded to primitive space by self.mlp (encoder part)
#             2. Random points sampled from 3D / hypersphere (decoder part)
#         '''
#         graph_protein_code = torch.repeat_interleave(output["graph_feature"], num_nodes, dim=0)
#         graph_protein_code = self.graph_representation_reduction(graph_protein_code)
#         # z_points = torch.cat([normalized_graph_points, graph_protein_code], dim=1)
#         # embedded_primitive_space_points = self.encoder(z_points)
        
#         # graph_protein_code_normalized = torch.div(output["graph_feature"].T, torch.tensor(self.num_nodes, device=graph.device)).T
#         # graph_protein_code_normalized = torch.repeat_interleave(graph_protein_code_normalized, num_nodes, dim=0)
#         random_primitive_space_points = self.sample_mesh_sphere(num_cum_nodes[-1], graph.device)
#         random_primitive_space_points = torch.cat([random_primitive_space_points, graph_protein_code], dim=1)
#         random_real_space_points = self.mapper(random_primitive_space_points)
#         # random_real_space_points = self.normalize_points(random_real_space_points)
#         random_real_space_points = torch.split(random_real_space_points, self.num_nodes)
        
#         return random_real_space_points, target

#     def evaluate(self, s_space_points, s_space_target):
#         metric = {}
        
#         # NOTE: Loss for reconstruction (s space), set distance between two point cloud
#         assert len(s_space_points) == len(s_space_target), "Number of data in prediction, target is different"
#         device = str(s_space_points[0].device)
#         # s_space_points = list(map(lambda x, y: x / y, s_space_points, self.num_nodes))
#         # s_space_points = list(map(lambda x : self.normalize_points(x), s_space_points))
#         s_space_distance = list(map(lambda x, y: self.point_cloud_distance(x, y, device), s_space_points, s_space_target))
#         s_space_distance = torch.tensor(s_space_distance, dtype=torch.float32, requires_grad=True).mean()
        
#         # NOTE: Loss for charformer distance (primitive space, u), distance from encoded points to mesh
#         # mesh_points = self.mesh_sphere.sample_points_uniformly(number_of_points=self.mesh_number_of_points)
#         # u_space_points = self.normalize_points(u_space_points)
#         # u_space_points = list(map(lambda x, y: x / y, u_space_points, self.num_nodes))
#         # u_space_points = torch.cat(u_space_points, dim=0)
#         # embedded_points = o3d.t.geometry.PointCloud(o3c.Tensor(u_space_points.tolist(), o3c.float32, device=o3c.Device(device))).to_legacy()
#         # u_space_distance = embedded_points.compute_point_cloud_distance(mesh_points)
#         # u_space_distance = torch.tensor(u_space_distance, dtype=torch.float32, requires_grad=True).mean()
#         '''
#         NOTE: mean of mean is different to just mean since the number of points are different
#         But probably won't make much difference
#         '''
#         # primitive_space_distance = torch.split(primitive_space_distance, self.num_nodes)
#         # primitive_space_distance = list(map(lambda x: torch.mean(x), primitive_space_distance))
#         # primitive_space_distance = torch.tensor(primitive_space_distance, dtype=torch.float32, requires_grad=True).mean()
        
#         name_s = tasks._get_metric_name("distance_3d")
#         # name_u = tasks._get_metric_name("distance_primitive")
#         metric[name_s] = s_space_distance
#         # metric[name_u] = u_space_distance

#         return metric, s_space_distance

#     def forward(self, batch):
#         """"""
#         all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
#         metric = {}

#         s_space_points, s_space_target = self.predict_and_target(batch, all_loss, metric)
#         metric_info, loss = self.evaluate(s_space_points, s_space_target)
#         metric.update(metric_info)

#         # loss = F.cross_entropy(pred, target)
#         name = tasks._get_criterion_name("distance_loss")
#         metric[name] = loss

#         all_loss += loss

#         return all_loss, metric



