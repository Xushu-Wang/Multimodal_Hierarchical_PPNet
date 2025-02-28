import torch
from yacs.config import CfgNode
from model.hierarchical import ProtoNode
from model.multimodal import CombinerProtoNode
from dataio.dataset import Mode
from typing import Union

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

    def __init__(self, mode, cfg_coef: CfgNode, N: int, epoch = "train"): 
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
    if len(target) == 0: 
        return torch.zeros(1, device=target.device), torch.zeros(1, device=target.device) 

    # Create one-hot encoding
    target_one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    # make_one_hot(target + 1, target_one_hot)
    num_prototypes_per_class = min_dist.size(1) // num_classes
    one_hot_repeat = target_one_hot.unsqueeze(2).repeat(1,1,num_prototypes_per_class).\
                        view(target_one_hot.size(0),-1)
    cluster_cost = torch.mean(torch.min(min_dist* one_hot_repeat, dim=1)[0])

    flipped_one_hot_repeat = 1 - one_hot_repeat
    inverted_distances_to_nontarget_prototypes, _ = \
        torch.max((0 - min_dist) * flipped_one_hot_repeat, dim=1)
    separation_cost = torch.mean(0 - inverted_distances_to_nontarget_prototypes)

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
    diff = sim_matrix(node.prototype) - torch.eye(node.prototype.shape[0]).cuda()
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

def get_loss_multi(
    conv_features,
    node: Union[ProtoNode, CombinerProtoNode],
    target: torch.Tensor
):
    """
    For parallel, no global ce
    """ 
    # create mask on target  
    mask = node.idx


    # Given the target of 80 labels, you only want to look at the relevant ones. 
    # If your node idx is [2, 3], then you should be looking for all samples of 
    # the form [2, 3, *], and you filter out only * 

    depth = node.gen_node.taxnode.depth

    if depth == 0: 
        # this is root node that classifies order and you should consider everything 
        mask = torch.ones(target.size(0), dtype=torch.bool)
    else: 
        # there might be irrelevant nodes and you should mask them 
        mask = (target[:,depth-1] == node.gen_node.taxnode.idx[-1])

    # TODO: actually to speed up mask the conv features first 
    
    (gen_logits, img_logits), (gen_min_dist, img_min_dist) = node(conv_features, True) 
    gen_logits = gen_logits[mask] # [B, nclasses] -> [M, nclasses] for B >= M 
    img_logits = img_logits[mask]
    
    target = target[mask] # [B, 4] -> [M, 4]
    gen_min_dist = gen_min_dist[mask] # [B, nproto_total] -> [M, nproto_total]
    img_min_dist = img_min_dist[mask] # [B, nproto_total] -> [M, nproto_total]
    conv_features = conv_features[mask] 


    if mask.sum() == 0: 
        # no samples are relevant for this node since they are not in the taxonomy
        return 0, 0, 0, 0, 0


    if hasattr(node.gen_node, "_logits"): 
        del node.gen_node._logits
    if hasattr(node.img_node, "_logits"): 
        del node.img_node._logits 
    node.gen_node._logits = gen_logits
    node.img_node._logits = img_logits

    cross_entropy = torch.nn.functional.cross_entropy(logits, target[:, node.taxnode.depth] - 1)

    cluster_cost, separation_cost = get_cluster_and_sep_cost(
        min_distances, target[:, node.taxnode.depth], logits.size(1)
    )

    l1_cost = get_l1_cost(node) 

    # Update correct and total counts
    _, predicted = torch.max(logits, dim=1) 


    n_classifications = 1
    # Now recurse on the new ones 
    for c_node in node.childs:

        c_cross_entropy, c_cluster_cost, c_separation_cost, c_l1_cost, c_n_classifications = get_loss(
            conv_features,
            c_node,
            target
        )

        cross_entropy += c_cross_entropy
        cluster_cost += c_cluster_cost 
        separation_cost += c_separation_cost
        l1_cost += c_l1_cost
        n_classifications += c_n_classifications

    del logits, min_distances

    return cross_entropy, cluster_cost, separation_cost, l1_cost, n_classificat
