import torch
from yacs.config import CfgNode
from model.hierarchical import ProtoNode
from model.multimodal import CombinerProtoNode
from dataio.dataset import Mode
from typing import Union
from typing_extensions import deprecated

run_mode = {1 : "genetic", 2 : "image", 3 : "multimodal"} 

class CorrectCount: 
    """
    Data Structure that keeps track of total correct classifications at each level. 
    Attributes: 
        N - length of dataset 
    """
    def __init__(self, N: int): 
        # Order, Family, Genus, Species classifications 
        self.counts = [0, 0, 0, 0] 
        self.N = N

    def __getitem__(self, depth): 
        return self.counts[depth]

    def __setitem__(self, depth, count): 
        self.counts[depth] = count 

    def __iadd__(self, other): 
        for i in range(len(self.counts)): 
            self.counts[i] += other.counts[i]
        return self

    def acc(self): 
        return [x / self.N for x in self.counts] 

class Objective: 
    """
    Wrapper class around different loss values 
    Attributes: 
        mode: the model mode 
        epoch - either "train" or "test"
        N - number of samples in dataset
    """
    def __init__(self, mode: Mode, cfg_coef: CfgNode, N: int, epoch = "train"): 
        self.mode = mode
        if self.mode == Mode.GENETIC: 
            cfg_coef = cfg_coef.GENETIC
        elif self.mode == Mode.IMAGE: 
            cfg_coef = cfg_coef.IMAGE 
        else: 
            raise ValueError("Not the correct mode for objective.")
        self.epoch = epoch
        self.N = N 

        self.cross_entropy = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1).cuda()
        self.separation = torch.zeros(1).cuda()
        self.lasso = torch.zeros(1).cuda()
        self.orthogonality = torch.zeros(1).cuda()
        self.n_next_correct = CorrectCount(N)
        self.n_cond_correct = CorrectCount(N)

        self.coef_ce = cfg_coef.CRS_ENT
        self.coef_clst = cfg_coef.CLST
        self.coef_sep = cfg_coef.SEP
        self.coef_l1 = cfg_coef.L1
        self.coef_ortho = cfg_coef.ORTHO

    def total(self): 
        """
        Linear combination of the components with their coefficient weights
        """
        ce = self.coef_ce * self.cross_entropy 
        clst = self.coef_clst * self.cluster 
        sep = self.coef_sep * self.separation
        lasso = self.coef_l1 * self.lasso 
        ortho = self.coef_ortho * self.orthogonality
        return ce + clst + sep + lasso + ortho 

    def __iadd__(self, other): 
        """
        Adds only loss terms together  
        """
        # TODO: gradients should be turned off for this
        self.cross_entropy += other.cross_entropy
        self.cluster += other.cluster
        self.separation += other.separation 
        self.lasso += other.lasso 
        self.orthogonality += other.orthogonality
        self.n_next_correct += other.n_next_correct
        self.n_cond_correct += other.n_cond_correct
        return self 

    def __itruediv__(self, const: int): 
        """
        Used to scale every loss term down. Usually when we take the mean. 
        """
        self.cross_entropy /= const
        self.cluster /= const
        self.separation /= const
        self.lasso /= const
        self.orthogonality /= const
        return self

    def to_dict(self): 
        next_accs = self.n_next_correct.acc()
        cond_accs = self.n_cond_correct.acc()
        out = {
            f"{run_mode[self.mode.value]}-{self.epoch}-cross-ent": self.cross_entropy, 
            f"{run_mode[self.mode.value]}-{self.epoch}-cluster": self.cluster, 
            f"{run_mode[self.mode.value]}-{self.epoch}-separation": self.separation,
            f"{run_mode[self.mode.value]}-{self.epoch}-lasso": self.lasso, 
            f"{run_mode[self.mode.value]}-{self.epoch}-orthogonality": self.orthogonality, 
            f"{run_mode[self.mode.value]}-{self.epoch}-next-acc-base": next_accs[0], 
            f"{run_mode[self.mode.value]}-{self.epoch}-next-acc-order": next_accs[1], 
            f"{run_mode[self.mode.value]}-{self.epoch}-next-acc-family": next_accs[2], 
            f"{run_mode[self.mode.value]}-{self.epoch}-next-acc-genus": next_accs[3],
            f"{run_mode[self.mode.value]}-{self.epoch}-cond-acc-base": cond_accs[0], 
            f"{run_mode[self.mode.value]}-{self.epoch}-cond-acc-order": cond_accs[1], 
            f"{run_mode[self.mode.value]}-{self.epoch}-cond-acc-family": cond_accs[2], 
            f"{run_mode[self.mode.value]}-{self.epoch}-cond-acc-genus": cond_accs[3]
        }
        return out 

    def __str__(self): 
        next_accs = self.n_next_correct.acc()
        cond_accs = self.n_cond_correct.acc()
        out = "" 
        out += f"{run_mode[self.mode.value]}-{self.epoch}-cross-ent     : {float(self.cross_entropy.item()):.5f}\n"
        out += f"{run_mode[self.mode.value]}-{self.epoch}-separation    : {float(self.separation.item()):.5f}\n"
        out += f"{run_mode[self.mode.value]}-{self.epoch}-cluster       : {float(self.cluster.item()):.5f}\n"
        out += f"{run_mode[self.mode.value]}-{self.epoch}-lasso         : {float(self.lasso.item()):.5f}\n"
        out += f"{run_mode[self.mode.value]}-{self.epoch}-orthogonality : {float(self.orthogonality.item()):.5f}\n"
        out += f"{run_mode[self.mode.value]}-{self.epoch}-next-acc      : {next_accs[0]:.4f}, {next_accs[1]:.4f}, {next_accs[2]:.4f}, {next_accs[3]:.4f}\n"
        out += f"{run_mode[self.mode.value]}-{self.epoch}-cond-acc      : {cond_accs[0]:.4f}, {cond_accs[1]:.4f}, {cond_accs[2]:.4f}, {cond_accs[3]:.4f}\n"
        return out
    
    def __repr__(self): 
        return self.__str__() 

    def clear(self): 
        self.cross_entropy = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1).cuda()
        self.separation = torch.zeros(1).cuda()
        self.lasso = torch.zeros(1).cuda()
        self.orthogonality = torch.zeros(1).cuda()
        self.n_next_correct = CorrectCount(self.N)
        self.n_cond_correct = CorrectCount(self.N)
        torch.cuda.empty_cache()

class MultiObjective: 
    """
    Wrapper class of two Objective objects used to calculuate loss of 
    Multimodal ProtoPNet
    """

    def __init__(self, mode, cfg_coef: CfgNode, N: int, epoch):
        self.mode = mode
        assert self.mode == Mode.MULTIMODAL
        self.N = N 
        self.epoch = epoch 
        self.gen_obj = Objective(Mode.GENETIC, cfg_coef, N, epoch)
        self.img_obj = Objective(Mode.IMAGE, cfg_coef, N, epoch)
        self.correspondence = torch.zeros(1).cuda()
        self.coef_corr = cfg_coef.CORRESPONDENCE

    def total(self): 
        corr = self.coef_corr * self.correspondence 
        return self.gen_obj.total() + self.img_obj.total() + corr

    def __iadd__(self, other): 
        self.gen_obj += other.gen_obj 
        self.img_obj += other.img_obj 
        self.correspondence += other.correspondence
        return self

    def __itruediv__(self, const: int): 
        self.gen_obj /= const 
        self.img_obj /= const 
        return self 

    def to_dict(self): 
        out = {} 
        out.update(self.gen_obj.to_dict())
        out.update(self.img_obj.to_dict()) 
        out[f"{run_mode[self.mode.value]}-{self.epoch}-correspondence"] = self.correspondence
        return out

    def __str__(self): 
        out = "" 
        out += str(self.gen_obj)
        out += str(self.img_obj)
        out += f"{run_mode[self.mode.value]}-{self.epoch}-correspondence: {float(self.correspondence.item()):.5f}"
        return out

    def __repr__(self): 
        return self.__str__()

    def clear(self): 
        self.gen_obj.clear()
        self.img_obj.clear() 
        self.correspondence = torch.zeros(1).cuda()
        torch.cuda.empty_cache()

def get_cluster_and_sep_cost(min_dist, target, num_classes): 
    """
    Get cluster and separation cost.  
    Before, mask the sizes should be
        min_dist    - the minimum distance from each prototype 
                         - IMG - (80 & mask, 10 * num_classes) - flattened with torch.view
                         - GEN - (80 & mask, 40 * num_classes)
        target      - the relevant samples. There are (80 & mask) of them. 
        num_classes - number of classes to be predicted by this node (e.g. node outputs 1...N classes)
        Note that each target (e.g. 20 targets) are in one of the num_classes class
    """

    if len(target) == 0: 
        return torch.zeros(1, device=target.device), torch.zeros(1, device=target.device)  

    # Create one-hot encoding
    target_one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    # make_one_hot(target + 1, target_one_hot)
    num_prototypes_per_class = min_dist.size(1) // num_classes
    prototypes_of_correct_class = target_one_hot.unsqueeze(2).repeat(1,1,num_prototypes_per_class).\
                        view(target_one_hot.size(0),-1)  
    
    # one_hot_repeat = (80 & mask, num_classes * protos_per_class)  
    # min_dist = (80 & mask, 10 * num_classes)  
    
    max_dist = 2  # should be 2 since distance lives in [0, 2]

    inverted_distances, _ = torch.max((max_dist - min_dist) * prototypes_of_correct_class, dim=1)
    cluster_cost = torch.mean(max_dist - inverted_distances)

    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    inverted_distances_to_nontarget_prototypes, _ = \
    torch.max((max_dist - min_dist) * prototypes_of_wrong_class, dim=1)
    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

    return cluster_cost, separation_cost

def get_l1_cost(node: ProtoNode):
    l1_mask = (1 - node.match).t().to("cuda")
    masked_weights = node.last_layer.weight.detach().clone() * l1_mask
    l1 = torch.linalg.vector_norm(masked_weights, ord=1)
    return l1

def sim_matrix(prototypes):
    prototypes_cur = prototypes.squeeze(-1).squeeze(-1)
    prototypes_normed = prototypes_cur / (prototypes_cur.norm(dim=-1, keepdim=True) + 1e-6)
    return prototypes_normed @ prototypes_normed.T

def get_ortho_cost(node: ProtoNode, temp=0.01):
    diff = sim_matrix(node.prototype) - torch.eye(node.prototype.size(0)).cuda()
    if temp is not None:
        mask = torch.nn.functional.softmax(diff / temp, dim=-1)
    else:
        mask = torch.ones_like(diff)

    return torch.norm(
        mask * diff
    )

def get_correspondence_loss_batched(
    gen_min_dist,
    img_min_dist,
    node
):
    if node.taxonomy == "Diptera":
        node.correlation_count += len(gen_min_dist)

    wrapped_genetic_min_distances = gen_min_dist.view(
        -1, gen_min_dist.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_min_distances_along_the_third_axis = img_min_dist.unsqueeze(2).expand(-1, -1, node.prototype_ratio)

    # Calculate the dot product of the normalized distances along the batch dimension (gross)
    l2_distance = (wrapped_genetic_min_distances - repeated_image_min_distances_along_the_third_axis) ** 2
    total_dist = torch.sum(
        l2_distance,
        dim=0
    )

    # Get the maximum dot product for each image prototype
    min_correspondence_costs, min_correspondence_cost_indicies = torch.min(total_dist, dim=1)

    correspondence_cost_count = len(min_correspondence_costs)
    correspondence_cost_summed = torch.sum(min_correspondence_costs)

    result = correspondence_cost_summed, correspondence_cost_count
    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis, l2_distance, total_dist, min_correspondence_costs, min_correspondence_cost_indicies

    torch.cuda.empty_cache()
    return result

def get_correspondence_loss_single(
    gen_min_dist,
    img_min_dist,
    mask,
    node
):
    wrapped_genetic_min_distances = gen_min_dist[mask].view(
        -1, gen_min_dist.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_min_distances_along_the_third_axis = img_min_dist[mask].unsqueeze(2).expand(-1, -1, node.prototype_ratio)
    
    # Calculate the total correspondence cost, minimum MSE between corresponding prototypes. We will later divide this by the number of comparisons made to get the average correspondence cost
    correspondence_cost_count = len(wrapped_genetic_min_distances)
    correspondence_cost_summed = torch.sum(
        torch.min(
            (wrapped_genetic_min_distances - repeated_image_min_distances_along_the_third_axis) ** 2,
            dim=2
        )[0]
    )

    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis

    return correspondence_cost_summed, correspondence_cost_count

