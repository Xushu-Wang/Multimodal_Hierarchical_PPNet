import torch
from model.hierarchical import ProtoNode
from model.multimodal import CombinerProtoNode
from dataio.dataset import Mode
from typing import Union

class Objective: 
    """
    Wrapper class around different loss values 
    """
    def __init__(self, cfg, epoch = "train"): 
        self.mode = Mode(cfg.DATASET.MODE)
        self.epoch = epoch

        self.cross_entropy = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1).cuda()
        self.separation = torch.zeros(1).cuda()
        self.lasso = torch.zeros(1).cuda()

        self.coef_ce = cfg.OPTIM.COEFS.CRS_ENT
        self.coef_clst = cfg.OPTIM.COEFS.CLST
        self.coef_sep = cfg.OPTIM.COEFS.SEP
        self.coef_l1 = cfg.OPTIM.COEFS.L1

        if self.mode == Mode.MULTIMODAL:
            self.gen_orthogonality = torch.zeros(1).cuda()
            self.img_orthogonality = torch.zeros(1).cuda()
            self.correspondence = torch.zeros(1).cuda()

            self.coef_gen_ortho = cfg.OPTIM.COEFS.ORTHOGONALITY.GENETIC
            self.coef_img_ortho = cfg.OPTIM.COEFS.ORTHOGONALITY.IMAGE
            self.coef_corr = cfg.OPTIM.COEFS.CORRESPONDENCE

    def total(self): 
        """
        Linear combination of the components with their coefficient weights
        """
        ce = self.coef_ce * self.cross_entropy 
        clst = self.coef_clst * self.cluster 
        sep = self.coef_sep * self.separation
        lasso = self.coef_l1 * self.lasso 

        if self.mode == Mode.MULTIMODAL: 
            gen_ortho = self.coef_gen_ortho * self.gen_orthogonality
            img_ortho = self.coef_img_ortho * self.img_orthogonality
            corr = self.coef_corr * self.correspondence
            return ce + clst + sep + lasso + gen_ortho + img_ortho + corr

        return ce + clst + sep + lasso

    def __iadd__(self, other): 
        # TODO: gradients should be turned off for this
        self.cross_entropy += other.cross_entropy
        self.cluster += other.cluster
        self.separation += other.separation 
        self.lasso += other.lasso 
        if self.mode == Mode.MULTIMODAL and other.mode == Mode.MULTIMODAL: 
            self.gen_orthogonality += other.gen_orthogonality
            self.img_orthogonality += other.img_orthogonality
            self.correspondence += other.correspondence
        return self 

    def __itruediv__(self, const: int): 
        """
        Used to scale everything down. Usually when we take the mean. 
        """
        self.cross_entropy /= const
        self.cluster /= const
        self.separation /= const
        self.lasso /= const
        if self.mode == Mode.MULTIMODAL: 
            self.gen_orthogonality /= const
            self.img_orthogonality /= const
            self.correspondence /= const
        return self

    def to_dict(self): 
        out = {
            f"{self.epoch}-cross_ent": self.cross_entropy, 
            f"{self.epoch}-cluster": self.cluster, 
            f"{self.epoch}-separation": self.separation,
            f"{self.epoch}-l1": self.lasso
        }
        if self.mode == Mode.MULTIMODAL: 
            out[f"{self.epoch}-gen_ortho"] = self.gen_orthogonality 
            out[f"{self.epoch}-img_ortho"] = self.img_orthogonality 
            out[f"{self.epoch}-corr"] = self.correspondence
        return out 

    def __str__(self): 
        out = "" 
        out += f"{self.epoch}-cross_ent: {float(self.cross_entropy.item()):.5f}\n"
        out += f"{self.epoch}-separation: {float(self.separation.item()):.5f}\n"
        out += f"{self.epoch}-cluster: {float(self.cluster.item()):.5f}\n"
        out += f"{self.epoch}-lasso: {float(self.lasso.item()):.5f}\n"
        if self.mode == Mode.MULTIMODAL: 
            out += f"{self.epoch}-gen_ortho: {float(self.gen_orthogonality.item()):.5f}\n"
            out += f"{self.epoch}-img_ortho: {float(self.img_orthogonality.item()):.5f}\n"
            out += f"{self.epoch}-corr: {float(self.correspondence.item()):.5f}\n"
        return out
    
    def __repr__(self): 
        return self.__str__()

def make_one_hot(target, target_one_hot):
    target_copy = torch.LongTensor(len(target))
    target_copy.copy_(target)
    target_copy = target_copy.view(-1,1).to("cuda")
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target_copy, value=1.)

def get_cluster_and_sep_cost(min_dist, target, num_classes): 
    if len(target) == 0: 
        return torch.zeros(1, device=target.device), torch.zeros(1, device=target.device)
    
    # Create one-hot encoding
    target_one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    make_one_hot(target, target_one_hot)
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

def get_ortho_cost(node: ProtoNode):
    P = node.prototype.squeeze(-1).squeeze(-1)  # type: ignore
    P = P / torch.linalg.norm(P, dim=1).unsqueeze(-1)
    return torch.sum(P@P.T-torch.eye(P.size(0)).cuda())

def get_correspondence_loss_batched(
    gen_min_dist,
    img_min_dist,
    node
):
    if node.taxonomy == "Diptera":
        # node.correlation_table += get_correlation_matrix(gen_min_dist, img_min_dist)
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

    individual_min_indicies = torch.min(
        l2_distance,
        dim=2
    )[1]

    node.max_tracker[0].append(min_correspondence_cost_indicies)
    node.max_tracker[1].append(individual_min_indicies)

    correspondence_cost_count = len(min_correspondence_costs)
    correspondence_cost_summed = torch.sum(min_correspondence_costs)

    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis, l2_distance, total_dist

    return correspondence_cost_summed, correspondence_cost_count

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

    node.npredictions += len(predicted)
    node.correct += (predicted == target[:, node.taxnode.depth] - 1).sum() 

    n_classifications = 1
    # Now recurse on the new ones 
    for _, c_node in node.childs.items():

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

    return cross_entropy, cluster_cost, separation_cost, l1_cost, n_classifications

