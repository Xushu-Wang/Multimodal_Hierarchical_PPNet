import torch
from yacs.config import CfgNode
from model.hierarchical import ProtoNode
from dataio.dataset import Mode

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
        self.cluster = torch.zeros(4).cuda()
        self.separation = torch.zeros(4).cuda()
        self.cluster_sep_count = torch.zeros(4).cuda()

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
        clst = self.coef_clst * (self.cluster.sum() / self.cluster_sep_count.sum()) 
        sep = self.coef_sep * (self.separation.sum() / self.cluster_sep_count.sum())
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
        self.cluster_sep_count += other.cluster_sep_count
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
            f"{self.epoch}/{run_mode[self.mode.value]}-cross-ent": self.cross_entropy, 
            f"{self.epoch}/{run_mode[self.mode.value]}-cluster": self.cluster.mean(), 
            f"{self.epoch}/{run_mode[self.mode.value]}-cluster-order": self.cluster[0], 
            f"{self.epoch}/{run_mode[self.mode.value]}-cluster-family": self.cluster[1], 
            f"{self.epoch}/{run_mode[self.mode.value]}-cluster-genus": self.cluster[2], 
            f"{self.epoch}/{run_mode[self.mode.value]}-cluster-species": self.cluster[3], 
            f"{self.epoch}/{run_mode[self.mode.value]}-separation": self.separation.mean(),
            f"{self.epoch}/{run_mode[self.mode.value]}-separation-order": self.separation[0],
            f"{self.epoch}/{run_mode[self.mode.value]}-separation-family": self.separation[1],
            f"{self.epoch}/{run_mode[self.mode.value]}-separation-genus": self.separation[2],
            f"{self.epoch}/{run_mode[self.mode.value]}-separation-species": self.separation[3],
            f"{self.epoch}/{run_mode[self.mode.value]}-lasso": self.lasso, 
            f"{self.epoch}/{run_mode[self.mode.value]}-orthogonality": self.orthogonality, 
            f"{self.epoch}/{run_mode[self.mode.value]}-next-acc-base": next_accs[0], 
            f"{self.epoch}/{run_mode[self.mode.value]}-next-acc-order": next_accs[1], 
            f"{self.epoch}/{run_mode[self.mode.value]}-next-acc-family": next_accs[2], 
            f"{self.epoch}/{run_mode[self.mode.value]}-next-acc-genus": next_accs[3],
            f"{self.epoch}/{run_mode[self.mode.value]}-cond-acc-base": cond_accs[0], 
            f"{self.epoch}/{run_mode[self.mode.value]}-cond-acc-order": cond_accs[1], 
            f"{self.epoch}/{run_mode[self.mode.value]}-cond-acc-family": cond_accs[2], 
            f"{self.epoch}/{run_mode[self.mode.value]}-cond-acc-genus": cond_accs[3]
        }
        return out 

    def __str__(self): 
        next_accs = self.n_next_correct.acc()
        cond_accs = self.n_cond_correct.acc()
        out = "" 
        out += f"{self.epoch}/{run_mode[self.mode.value]}-cross-ent     : {float(self.cross_entropy.item()):.5f}\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-separation-individual  : {float(self.separation[0].item()):.5f}, {float(self.separation[1].item()):.5f}, {float(self.separation[2].item()):.5f}, {float(self.separation[3].item()):.5f})\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-separation    : {float(self.separation.mean().item()):.5f}\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-cluster-individual  : {float(self.cluster[0].item()):.5f}, {float(self.cluster[1].item()):.5f}, {float(self.cluster[2].item()):.5f}, {float(self.cluster[3].item()):.5f})\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-cluster       : {float(self.cluster.mean().item()):.5f}\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-lasso         : {float(self.lasso.item()):.5f}\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-orthogonality : {float(self.orthogonality.item()):.5f}\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-next-acc      : {next_accs[0]:.4f}, {next_accs[1]:.4f}, {next_accs[2]:.4f}, {next_accs[3]:.4f}\n"
        out += f"{self.epoch}/{run_mode[self.mode.value]}-cond-acc      : {cond_accs[0]:.4f}, {cond_accs[1]:.4f}, {cond_accs[2]:.4f}, {cond_accs[3]:.4f}\n"
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
        out[f"{self.epoch}/{run_mode[self.mode.value]}-correspondence"] = self.correspondence
        return out

    def __str__(self): 
        out = "" 
        out += str(self.gen_obj)
        out += str(self.img_obj)
        out += f"{self.epoch}/{run_mode[self.mode.value]}-correspondence: {float(self.correspondence.item()):.5f}"
        return out

    def __repr__(self): 
        return self.__str__()

    def clear(self): 
        self.gen_obj.clear()
        self.img_obj.clear() 
        self.correspondence = torch.zeros(1).cuda()
        torch.cuda.empty_cache()

def get_cluster_and_sep_cost(max_sim, target, num_classes): 
    """
    Get cluster and separation cost.  
    Before, mask the sizes should be
        max_sim    - the minimum similarity with each prototype 
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
    num_prototypes_per_class = max_sim.size(1) // num_classes
    prototypes_of_correct_class = target_one_hot.unsqueeze(2).repeat(1,1,num_prototypes_per_class).\
                        view(target_one_hot.size(0),-1)  
    
    max_max_sims, _ = torch.max(max_sim * prototypes_of_correct_class, dim=1)
    cluster_cost = torch.sum(max_max_sims)

    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    max_max_sims_to_nontarget_prototypes, _ = torch.max(max_sim * prototypes_of_wrong_class, dim=1)
    separation_cost = torch.sum(max_max_sims_to_nontarget_prototypes)

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
    gen_max_sim,
    img_max_sim,
    node
):
    if node.taxonomy == "Diptera":
        node.correlation_count += len(gen_max_sim)

    wrapped_genetic_max_sim = gen_max_sim.view(
        -1, gen_max_sim.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_max_sim_along_the_third_axis = img_max_sim.unsqueeze(2).expand(-1, -1, node.prototype_ratio)

    # Calculate the dot product of the normalized distances along the batch dimension (gross)
    l2_distance = (wrapped_genetic_max_sim - repeated_image_max_sim_along_the_third_axis) ** 2
    total_dist = torch.sum(
        l2_distance,
        dim=0
    )

    # Get the maximum dot product for each image prototype
    min_correspondence_costs, min_correspondence_cost_indicies = torch.min(total_dist, dim=1)

    correspondence_cost_count = len(min_correspondence_costs)
    correspondence_cost_summed = torch.sum(min_correspondence_costs)

    result = correspondence_cost_summed, correspondence_cost_count
    del wrapped_genetic_max_sim, repeated_image_max_sim_along_the_third_axis, l2_distance, total_dist, min_correspondence_costs, min_correspondence_cost_indicies

    torch.cuda.empty_cache()
    return result

def get_correspondence_loss_single(
    gen_max_sim,
    img_max_sim,
    mask,
    node
):
    wrapped_genetic_max_sim = gen_max_sim[mask].view(
        -1, gen_max_sim.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_max_sim_along_the_third_axis = img_max_sim[mask].unsqueeze(2).expand(-1, -1, node.prototype_ratio)
    
    # Calculate the total correspondence cost, minimum MSE between corresponding prototypes. We will later divide this by the number of comparisons made to get the average correspondence cost
    correspondence_cost_count = len(wrapped_genetic_max_sim)
    correspondence_cost_summed = torch.sum(
        torch.min(
            (wrapped_genetic_max_sim - repeated_image_max_sim_along_the_third_axis) ** 2,
            dim=2
        )[0]
    )

    del wrapped_genetic_max_sim, repeated_image_max_sim_along_the_third_axis

    return correspondence_cost_summed, correspondence_cost_count

