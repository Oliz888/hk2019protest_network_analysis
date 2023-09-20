import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import imageio.v2 as imageio
import matplotlib.animation as animation

from collections import Counter
from scipy.stats import linregress
import random


from scipy import stats
import collections



#------------------------------
# ANIMATION PLOT
#------------------------------
def animate(networks, pos_type="circular", path="output.gif", plot_every=1):
    # check valid pos_type
    if pos_type not in ["circular", "random", "spring"]:
        raise ValueError("Invalid pos_type. Expected one of: %s" % ["circular", "random", "spring"])

    file_names = []  # to store temporary file names
    for i, G in enumerate(networks[::plot_every], start=1):
        # determine layout
        if pos_type == "circular":
            pos = nx.circular_layout(G)
        elif pos_type == "random":
            pos = nx.random_layout(G)
        else:  # "spring"
            pos = nx.spring_layout(G)

        # Initialize figure
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)

        # Draw bounding box
        tmpx = []; tmpy = []
        for p in pos.values():
            tmpx.append(p[0])
            tmpy.append(p[1])
        Lxmin = min(tmpx) - 0.2; Lxmax = max(tmpx) + 0.2
        Lymin = min(tmpy) - 0.2; Lymax = max(tmpy) + 0.2
        ax.axhline(y=Lymin); ax.axvline(x=Lxmin)
        ax.axhline(y=Lymax); ax.axvline(x=Lxmax)

        # Draw graph
        if nx.is_directed(G):
            node_sizes = [degree for node, degree in G.in_degree()]
            node_colors = [degree for node, degree in G.out_degree()]
        else:
            node_degrees = [degree for node, degree in G.degree()]
            node_sizes = node_colors = node_degrees
        nx.draw(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)

        # Save figure to a temporary png file
        temp_file_name = f"tmp-{i}.png"
        file_names.append(temp_file_name)
        plt.savefig(temp_file_name)
        plt.close()

    # Create gif
    images = list(map(lambda filename: imageio.imread(filename), file_names))
    imageio.mimsave(path, images, duration = 10/len(file_names))

    # Clean up
    for file_name in file_names:
        os.remove(file_name)
    
#------------------------------
# NETWORK CENTRALITY CORRELATION PLOTS
#------------------------------
def plot_centrality_correlation(G, path=""):
    if nx.is_directed(G):
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)
        in_closeness_centrality = nx.closeness_centrality(G)
        out_closeness_centrality = nx.closeness_centrality(G.reverse())
        betweenness_centrality = nx.betweenness_centrality(G.to_undirected())

        data = pd.DataFrame({
            'In Degree': list(in_degree_centrality.values()),
            'Out Degree': list(out_degree_centrality.values()),
            'In Closeness': list(in_closeness_centrality.values()),
            'Out Closeness': list(out_closeness_centrality.values()),
            'Betweenness': list(betweenness_centrality.values())
        })
    else:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        data = pd.DataFrame({
            'Degree': list(degree_centrality.values()),
            'Betweenness': list(betweenness_centrality.values()),
            'Closeness': list(closeness_centrality.values())
        })

    sns.pairplot(data, diag_kind="hist")

    if path != "":
        plt.savefig(path)
    
    plt.show()

#------------------------------
# AVERAGE DEGREE
#------------------------------
def ave_degree(G):
    if nx.is_directed(G):
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]

        ave_in_degree = sum(in_degrees) / float(G.number_of_nodes())
        ave_out_degree = sum(out_degrees) / float(G.number_of_nodes())
        
        print(f'Average in-degree: {ave_in_degree:.2f}')
        print(f'Average out-degree: {ave_out_degree:.2f}')

    else:
        degrees = [d for n, d in G.degree()]
        ave_degree = sum(degrees) / float(G.number_of_nodes())
        print(f'Average degree: {ave_degree:.2f}')

#------------------------------
# PLOT DEGREE DISTRIBUTION
#------------------------------

def plot_degree_distribution(G, type="", fit=False, path=""):
    # Create 3*3 plots
    fig, axs = plt.subplots(1, 4, figsize=(8, 4))

    # Calcualte degree
    degrees = dict(G.degree())
    #degree_values = list(degrees.values())
    degree_vals = list(filter(lambda val: val > 0, degrees.values()))
    # getting unique and sorted outdegree values
    uq_degree_vals = sorted(set(degree_vals))
    # counting frequency of each outdegree values
    out_hist = [degree_vals.count(x) for x in uq_degree_vals]
    # Plot PDF of degree
    sns.histplot(degree_vals, ax=axs[0], stat="density")
    axs[0].set_xlabel('Degree')
    axs[0].set_ylabel('Probability')
    # Plot log PDF of degree
    x = np.asarray(uq_degree_vals, dtype = float)
    y = np.asarray(out_hist, dtype = float)

    logx = np.log10(x)
    logy = np.log10(y)

    axs[1].plot(logx, logy, 'o')
    #counts_degree,bins_degree=np.histogram(degree_values, bins=50,density=False)
    #bins_degree=(np.array(bins_degree[1:])+np.array(bins_degree[0:-1]))/2.0
    #axs[1].plot(bins_degree,counts_degree/len(G.nodes())/(bins_degree[1]-bins_degree[0]),"o")
    #axs[1].set_xscale('log')
    #axs[1].set_yscale('log') 
    axs[1].set_xlabel('Degree')
    axs[1].set_ylabel('Probability(log)')
    # Plot cCDF of degree
    sns.ecdfplot(degree_vals, ax=axs[2],linewidth=5, complementary=True)
    axs[2].set_xlabel('Degree')
    axs[2].set_ylabel('cCDF')
    # Plot log cCDF of degree
    sns.ecdfplot(degree_vals, ax=axs[3],linewidth=5, complementary=True)
    axs[3].set_xlabel('Degree')
    axs[3].set_ylabel('cCDF(log)')
    axs[3].set_xscale('log')
    axs[3].set_yscale('log')

    if fit:
        a, b = np.polyfit(logx, logy, 1)
        #axs[1].plot(logx, logy, 'o')
        axs[1].plot(logx, a*logx + b)
        
        ccdf_values = np.cumsum(out_hist[::-1])[::-1] / len(degree_vals)
        degree_values = np.asarray(uq_degree_vals, dtype = float)
        log_degree_values = np.log10(degree_values)
        log_ccdf_values = np.log10(ccdf_values)
        #rank_values = np.arange(1, len(degree_vals) + 1)
        #ccdf_values = 1.0 - rank_values / len(degree_vals)
        #ccdf_values_nonzero = ccdf_values[ccdf_values > 0]  # Exclude zero or negative values
        #log_ccdf_values = np.log(ccdf_values_nonzero)
        #log_degree_values = np.asarray(uq_degree_vals, dtype = float)
        a, b = np.polyfit(log_degree_values, log_ccdf_values, 1)
        #axs[3].plot(log_degree_values, log_ccdf_values, 'o')
        axs[3].plot(log_degree_values, a * log_degree_values+b)
        
    plt.tight_layout()
    plt.show()

    # Save the plot to the path if provided
    if path != "":
        fig.savefig(path)


#------------------------------
# NETWORK PLOTTING FUNCTION
#------------------------------
def plot_network(G,node_color="degree",layout="random"):
    
    # POSITIONS LAYOUT
    N=len(G.nodes)
    if(layout=="spring"):
        # pos=nx.spring_layout(G,k=50*1./np.sqrt(N),iterations=100)
        pos=nx.spring_layout(G)

    if(layout=="random"):
        pos=nx.random_layout(G)

    #INITALIZE PLOT
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    # NODE COLORS
    cmap=plt.cm.get_cmap('Greens')

    # DEGREE 
    if node_color=="degree":
            centrality=list(dict(nx.degree(G)).values())
  
    # BETWENNESS 
    if node_color=="betweeness":
            centrality=list(dict(nx.betweenness_centrality(G)).values())
  
    # CLOSENESS
    if node_color=="closeness":
            centrality=list(dict(nx.closeness_centrality(G)).values())

    # NODE SIZE CAN COLOR
    node_colors = [cmap(u/(0.01+max(centrality))) for u in centrality]
    node_sizes = [4000*u/(0.01+max(centrality)) for u in centrality]

    # #PLOT NETWORK
    nx.draw(G,
            with_labels=True,
            edgecolors="black",
            node_color=node_colors,
            node_size=node_sizes,
            font_color='white',
            font_size=18,
            pos=pos
            )

    plt.show()

#------------------------------
# NETWORK SUMMARY FUNCTION
#------------------------------
def network_summary(G):

    def centrality_stats(x):
        x1=dict(x)
        x2=np.array(list(x1.values())); #print(x2)
        print("	min:" ,min(x2))
        print("	mean:" ,np.mean(x2))
        print("	median:" ,np.median(x2))
        # print("	mode:" ,stats.mode(x2)[0][0])
        print("	max:" ,max(x2))
        x=dict(x)
        sort_dict=dict(sorted(x1.items(), key=lambda item: item[1],reverse=True))
        print("	top nodes:",list(sort_dict)[0:6])
        print("	          ",list(sort_dict.values())[0:6])

    try: 
        print("GENERAL")
        print("	number of nodes:",len(list(G.nodes)))
        print("	number of edges:",len(list(G.edges)))

        print("	is_directed:", nx.is_directed(G))
        print("	is_weighted:" ,nx.is_weighted(G))


        if(nx.is_directed(G)):
            print("IN-DEGREE (NORMALIZED)")
            centrality_stats(nx.in_degree_centrality(G))
            print("OUT-DEGREE (NORMALIZED)")
            centrality_stats(nx.out_degree_centrality(G))
        else:
            print("	number_connected_components", nx.number_connected_components(G))
            print("	number of triangle: ",len(nx.triangles(G).keys()))
            print("	density:" ,nx.density(G))
            print("	average_clustering coefficient: ", nx.average_clustering(G))
            print("	degree_assortativity_coefficient: ", nx.degree_assortativity_coefficient(G))
            print("	is_tree:" ,nx.is_tree(G))

            if(nx.is_connected(G)):
                print("	diameter:" ,nx.diameter(G))
                print("	radius:" ,nx.radius(G))
                print("	average_shortest_path_length: ", nx.average_shortest_path_length(G))

            #CENTRALITY 
            print("DEGREE (NORMALIZED)")
            centrality_stats(nx.degree_centrality(G))

            print("CLOSENESS CENTRALITY")
            centrality_stats(nx.closeness_centrality(G))

            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G))
    except:
        print("unable to run")

#------------------------------
# ISOLATE GCC
#------------------------------
def isolate_GCC(G):
    comps = sorted(nx.connected_components (G),key=len, reverse=True) 
    nodes_in_giant_comp = comps[0]
    return nx.subgraph(G, nodes_in_giant_comp)


