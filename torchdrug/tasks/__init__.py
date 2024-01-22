from .task import Task

from .property_prediction import PropertyPrediction, MultipleBinaryClassification, \
    NodePropertyPrediction, InteractionPrediction, Unsupervised, UnsupervisedTwo
from .pretrain import EdgePrediction, AttributeMasking, ContextPrediction, DistancePrediction, \
    AnglePrediction, DihedralPrediction
from .generation import AutoregressiveGeneration, GCPNGeneration
from .retrosynthesis import CenterIdentification, SynthonCompletion, Retrosynthesis
from .reasoning import KnowledgeGraphCompletion
from .contact_prediction import ContactPrediction

from .pretrain_point import AttributeMaskingByPoints, AttributeMaskingByPointsFiltered, \
    AttributeMaskingByPointsSet, ConfidenceScoreByPoints,\
    DenoisingStructure, RandomNoiseMatching, UniformNoiseMatching


_criterion_name = {
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "bce": "binary cross entropy",
    "ce": "cross entropy",
    "pcd": "point cloud distance",
    "distance_loss": "total distance loss"
}

_metric_name = {
    "mae": "mean absolute error",
    "mse": "mean squared error",
    "rmse": "root mean squared error",
    "acc": "accuracy",
    "mcc": "matthews correlation coefficient",
    "pcd": "point cloud distance",
    "sd": "set distance",
    # "distance_charformer": "distance in embedded space"
}


def _get_criterion_name(criterion):
    if criterion in _criterion_name:
        return _criterion_name[criterion]
    return "%s loss" % criterion


def _get_metric_name(metric):
    if metric in _metric_name:
        return _metric_name[metric]
    return metric


__all__ = [
    "PropertyPrediction", "MultipleBinaryClassification", "NodePropertyPrediction", "InteractionPrediction",
    "Unsupervised", "UnsupervisedTwo"
    "EdgePrediction","AttributeMasking",
    "ContextPrediction", "DistancePrediction", "AnglePrediction",
    "DihedralPrediction",
    "AutoregressiveGeneration", "GCPNGeneration",
    "CenterIdentification", "SynthonCompletion", "Retrosynthesis",
    "KnowledgeGraphCompletion",
    "ContactPrediction",
    
    "AttributeMaskingByPoints", "AttributeMaskingByPointsFiltered", "AttributeMaskingByPointsSet"
    "ConfidenceScoreByPoints",
    "DenoisingStructure", "ProteinWorkshopDenoising",
    "RandomNoiseMatching", "UniformNoiseMatching"
]