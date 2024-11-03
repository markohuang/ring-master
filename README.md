# RingMaster - a motif-based diffusion model for molecular generation
A working pytorch geometric implementation of [Hierarchical Generation of Molecular Graphs using Structural Motifs](https://github.com/wengong-jin/hgraph2graph) but using a diffusion model. A work in progress for improving performance. 

# molecular generation through motifs
As opposed to the smiles representation where an atom-by-atom approach is taken to create a molecular graph, RingMaster adopts motifs as the basic building blocks for molecular generation. This allows for a more structured approach to molecular generation and allows for the generation of more complex molecules.

Below are gifs showing the construction process of molecules given correctly diffused motifs and their orderings. 
<table>
  <tr>
    <td><img src="https://github.com/markohuang/selfies_diffusion/blob/master/gifs/output1.gif" alt="GIF 1" width="300"></td>
    <td><img src="https://github.com/markohuang/selfies_diffusion/blob/master/gifs/output2.gif" alt="GIF 2" width="300"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/markohuang/selfies_diffusion/blob/master/gifs/output3.gif" alt="GIF 3" width="300"></td>
    <td><img src="https://github.com/markohuang/selfies_diffusion/blob/master/gifs/output4.gif" alt="GIF 4" width="300"></td>
  </tr>
</table>
