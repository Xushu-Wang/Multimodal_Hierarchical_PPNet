import numpy as np
import torch 

class Node:
    def __init__(self, name, parent = None, label = None):
        self.parent = parent
        self.children = []
        self.children_to_labels = {}
        self.name = name
        self.label = label
        
    def add_children(self, names, labels = None):
        if not names:
            return
        
        if type(names) is not list:
            names = [names]
        if labels is None:
            labels = [i for i in range(len(self.children),len(self.children)+len(names))]
        names.sort()
        for i in range(len(names)):
            self.children.append(Node(names[i], parent=self, label = labels[i]))    
            self.children_to_labels.update({names[i] : labels[i]})
    
    def get_node(self, name):
        active_nodes = [self]
        
        while active_nodes:
            node = active_nodes.pop(0)
            if node.name == name:
                return node
            active_nodes.extend(node.children)
        
        print(f"Node for {name} not found")
        
        return None
    
    def get_children_names(self):
        return ([child.name for child in self.children])    

    def get_node_attr(self,name,attr):
        node = self.get_node(name)
        return getattr(node,attr)

    def set_node_attr(self,name,attr,value):
        node = self.get_node(name)
        return setattr(node,attr,value)        
                
    def num_children(self):
        return(len(self.children))

    def class_to_num_children(self):
        class_to_num = {}
        active_nodes = [self]
        while active_nodes:
            for node in active_nodes:
                class_to_num.update({node.name : node.num_children()})
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return class_to_num

    def class_to_proto_shape(self, x_per_child = 1, dimension = 512):
        class_to_shape = {}
        active_nodes = [self]
        while active_nodes:
            for node in active_nodes:
                if node.num_children() > 0:
                    class_to_shape.update({node.name : (x_per_child * node.num_children(),dimension,1,1)})
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return class_to_shape        

    def classes_with_children(self):
        classes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:
                    classes.append(node.name)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return classes

    def nodes_with_children(self):        
        nodes = []
        active_nodes = [self]
        while active_nodes:
            for node in active_nodes:
                if node.num_children() > 0:
                    nodes.append(node)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return nodes        

    def parents(self):
        ancestors = []
        ancestor = self.parent
        ancestors += ancestor
        
        while ancestor.parent is not None:
            ancestor = ancestor.parent
            ancestors += ancestor
            
        return ancestors
    
    def assign_descendents(self):
        active_nodes = [self.children]
        descendents = set()
        while active_nodes:
            node = active_nodes.pop()
            
            if isinstance(node, Node):
                descendents.add(node.name)
                
                for child in node.children:
                    active_nodes.append(child)
                  
        self.descendents = descendents

    def assign_all_descendents(self):
        active_nodes = [self]
        while active_nodes:
            node = active_nodes.pop()
            
            if isinstance(node, Node):
                node.assign_descendents()
                
                for child in node.children:
                    active_nodes.append(child)

    def closest_descendent_for(self,name):
        if name in self.children_names(): 
            return self.get_node(name)
        else:
            return [child for child in self.children if name in child.descendents][0]
     
    def has_logits(self):
        return self.num_children() > 1

    def get_distribution(self):                
        if self.has_logits():
            return torch.nn.functional.softmax(self.logits,1)
        else:
            batch_size = self.logits.size(0)
            return torch.ones((batch_size,1))

        
    def distribution_over_furthest_descendents(self,batch_size):
        if not self.has_logits():
            return torch.ones(batch_size,1).cuda()
        else:
            return torch.cat([torch.nn.functional.softmax(self.logits,1)[:,i].view(batch_size,1) * self.children[i].distribution_over_furthest_descendents(batch_size) for i in range(self.num_children())],1)            

    def names_of_joint_distribution(self):
        if self.num_children() == 1:
            return [self.children[0].name]
        elif self.num_children() == 0:
            return [self.name]
        else:
            return [child.names_of_joint_distribution() for child in self.children]


    def unwrap_names_of_joint(self,names):
        new_list = []
        
        def _unwrap(names, new_list):
            for item in names:
                if isinstance(item, list):
                    _unwrap(item, new_list)
                else:
                    new_list.append(item)
                    
        _unwrap(names, new_list)
             
        return new_list


    def assign_unif_distributions(self):
        for node in self.nodes_with_children():
            node.unif = (torch.ones(node.num_children()) / node.num_children()).cuda()

    def assign_proto_dirs(self):
        for node in self.nodes_with_children():
            node.proto_dir = node.name + "_prototypes"

    def get_leaf_nodes(self):
        nodes = []
        active_nodes = [self]
        
        while active_nodes:
            node = active_nodes.pop()
            if not node.children:
                nodes.append(node)
            else:
                active_nodes.extend(node.children)
                            
        return nodes

    def __str__(self):
        return "Node for " + self.name



    

