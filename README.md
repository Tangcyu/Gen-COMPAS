<!DOCTYPE html>
<html lang="en">
<body>

<h1>Gen-COMPAS: Generative committor-guided path sampling for rare events </h1>

<p align="center">
<img src="figures/scheme.png" alt="Gen-COMPAS workflow" width="500">
</p>

<p>
This repository provides a modular pipeline for protein structure generation, committor analysis, clustering, and trajectory reweighting.
It combines <strong>diffusion models</strong> for structure generation with <strong>Variational Committor Networks (VCN)</strong> for reaction coordinate learning, along with postprocessing tools for clustering, occupancy analysis, and trajectory reweighting.
</p>

<p>
For computing statistical weights and estimating free energy landscapes, please refer to the <strong>Riteweight</strong> method described in:<a href="https://www.pnas.org/doi/10.1073/pnas.2529246123">https://www.pnas.org/doi/10.1073/pnas.2529246123</a> or <a href="https://arxiv.org/html/2401.05597v1">https://arxiv.org/html/2401.05597v1</a>.
</p>

<p>The RiteWeight implementation and its downstream weighted FEL projection are available as stages of <code>workflow.py</code>.</p>

<h2>Overview</h2>

<ul>
<li>The main entry point is a single Python script that dispatches different stages of the workflow based on a YAML configuration file.</li>
<li><code>workflow.py</code> manages cumulative iteration directories and the distinct bootstrap/committor-guided schedules.</li>
<li>You can train models, perform inference, analyze trajectories, and perform reweighting with a unified interface.</li>
</ul>

<h2>Workflow Steps</h2>

<table>
<thead>
<tr>
<th>Step Name</th>
<th>Function</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>train_diffusion</code></td>
<td><code>train_diffusion_model()</code></td>
<td>Train a diffusion model for protein structure generation.</td>
</tr>
<tr>
<td><code>sample_diffusion</code></td>
<td><code>run_diffusion_inference()</code></td>
<td>Generate new protein conformations using a trained diffusion model.</td>
</tr>
<tr>
<td><code>train_committor</code></td>
<td><code>train_committor_model()</code></td>
<td>Train a Variational Committor Network (VCN) to predict committor probabilities from MD data.</td>
</tr>
<tr>
<td><code>committor_slice</code></td>
<td><code>run_committor_slice()</code></td>
<td>Slice generated structures around the committor transition region.</td>
</tr>
<tr>
<td><code>clustering</code></td>
<td><code>run_clustering()</code></td>
<td>Cluster conformations using k-means and extract representative structures.</td>
</tr>
<tr>
<td><code>occupancy</code></td>
<td><code>add_occupancy()</code></td>
<td>Add hydrogen atoms and set occupancy flags in PDB files for visualization or targeted MD.</td>
</tr>
<tr>
<td><code>namd</code></td>
<td><code>run_namd_workflow()</code></td>
<td>Run parallel targeted MD followed by unbiased MD with CPU or GPU NAMD commands.</td>
</tr>
<tr>
<td><code>riteweight</code></td>
<td><code>run_riteweight()</code></td>
<td>Compute trajectory weights and emit VCN data plus an RMSD-aligned diffusion training trajectory.</td>
</tr>
<tr>
<td><code>fel_estimate</code></td>
<td><code>run_fel_estimate()</code></td>
<td>Project RiteWeight results into weighted 1D or 2D free-energy landscapes.</td>
</tr>
</tbody>
</table>

<h2>Workflow Usage</h2>

<p><strong>Important for new systems:</strong> Use Gen-COMPAS stepwise when setting up a new system. Do not run the full workflow blindly to generate trajectories. Instead, inspect the configuration and dry-run schedule first, then run and check each stage before continuing. In particular, verify generated structures, committor slices, clustering/target selection, TMD behavior, and trajectory diagnostics. This helps catch generation or simulation errors caused by unsuitable parameter choices before they propagate to later calculations.</p>

<h3>Quick Start</h3>

<p>After installation, <code>gen-compas</code> is the primary command. Always inspect the resolved schedule and paths before starting an expensive run:</p>

<pre><code>gen-compas --config /path/to/workflow.yaml --iteration 0 1 2 --dry-run
gen-compas --config /path/to/workflow.yaml --iteration 0 1 2
</code></pre>

<p>A source checkout can also be run without installation. Direct execution works from a different working directory:</p>

<pre><code>python /path/to/Gen-COMPAS/workflow.py \
  --config ./workflow.yaml \
  --iteration 0 1 2
</code></pre>

<p>Relative paths in the YAML file are interpreted from the directory where the command is launched. For reproducible runs, prefer absolute paths for external trajectories, topologies, NAMD templates, and executables.</p>

<h3>Iteration Schedules and Data Flow</h3>

<pre><code>Iteration 0:
  train_diffusion -&gt; sample_diffusion -&gt; clustering -&gt; occupancy
  -&gt; namd -&gt; riteweight -&gt; fel_estimate

Iterations 1+:
  train_diffusion -&gt; train_committor -&gt; sample_diffusion
  -&gt; committor_slice -&gt; occupancy -&gt; namd -&gt; riteweight
  -&gt; fel_estimate
</code></pre>

<p>Iteration 0 trains diffusion from <code>Workflow.initial_diffusion_data</code>, generates and clusters targets, runs TMD followed by unbiased simulations, and produces the first RiteWeight result. Iterations 1 and later use the preceding RiteWeight outputs as diffusion and VCN training data. Their committor-slice stage retains generated frames within <code>0.5 +/- VCN.q_variance</code>, then selects <code>VCN.n_targets</code> number of candidates.</p>

<ul>
<li><code>Workflow.run_initial_unbiased: true</code> prepends <code>initial_unbiased</code> to iteration 0. The <code>initial_unbiased_template</code> files start directly from the configured A/B basin states; these trajectories are added to cumulative RiteWeight input but do not replace <code>initial_diffusion_data</code>.</li>
<li><code>Workflow.run_fel: false</code> removes <code>fel_estimate</code> from each schedule.</li>
<li><code>Workflow.warm_start_diffusion: true</code> initializes iteration N diffusion weights from iteration N-1 <code>best_model.pt</code>. It does not restore optimizer, scheduler, or epoch state.</li>
<li><code>Workflow.isolate_steps: true</code> runs regular stages in clean Python child processes, releasing GPU/native-library state between stages.</li>
</ul>

<p>The requested iteration numbers are the stopping control because no system-independent convergence criterion exists in the code. Inspect the committor slice, pathway/TMD diagnostics, and cumulative FEL before requesting another iteration.</p>

<h3>Controlling a Run</h3>

<pre><code># Continue after an interruption: skip completed stages and restart the failed/running stage
gen-compas --config workflow.yaml --iteration 0 1 2 --resume

# Pause after every successful stage; Enter continues and q stops cleanly
gen-compas --config workflow.yaml --iteration 0 1 2 --stepwise

# Run an inclusive subsection of an iteration
gen-compas --config workflow.yaml --iteration 1 --start-at sample_diffusion
gen-compas --config workflow.yaml --iteration 1 --start-at occupancy --stop-after namd

# Run one incomplete stage, or force a completed stage to run again
gen-compas --config workflow.yaml --iteration 1 --run_step sample_diffusion
gen-compas --config workflow.yaml --iteration 1 --rerun_step sample_diffusion
</code></pre>

<p>Each iteration writes an effective configuration and manifest below <code>Workflow.root_dir</code>. If the process is interrupted, rerun the same iteration list with <code>--resume</code>; completed steps are skipped and the interrupted step is restarted from the beginning of that stage. The manifest is replaced atomically so it remains readable even if the process stops while its status is being updated.</p>

<p>With <code>--stepwise</code>, the workflow pauses after every successfully completed stage so its output can be inspected. Press Enter to continue, or enter <code>q</code> to stop cleanly. A stopped stepwise run can be continued with the same iteration list plus <code>--resume --stepwise</code>.</p>

<p><code>--run_step STEP</code> runs exactly one incomplete stage and skips it if the manifest already says <code>completed</code>. <code>--rerun_step STEP</code> forces that stage to run again and increments its attempt count. Neither command automatically reruns downstream stages. Both can be combined with <code>--stepwise</code> or <code>--dry-run</code>, but not with <code>--start-at</code> or <code>--stop-after</code>. The selected stage must belong to every requested iteration; for example, committor stages cannot run in iteration 0.</p>

<p>Run <code>gen-compas --help</code> to print every accepted step name and the iteration-specific schedules.</p>

<p>Sampling noise and diffusion-training epochs can be changed by iteration with sparse overrides. Omitted iterations retain <code>Generative.inference.noise_scale</code> and <code>Generative.training.epochs</code>, respectively:</p>

<pre><code>Workflow:
  root_dir: ./Iterations
  warm_start_diffusion: true
  isolate_steps: true
  iteration_noise_scales:
    0: 10.0
    1: 5.0
    2: 1.5
  iteration_diffusion_epochs:
    0: 50
    1: 10
    2: 10

VCN:
  q_variance: 0.1
  n_targets: 20
  require_n_targets: true
  cvs_to_plot: [CV1, CV2]
  plot_committor_projections: false
</code></pre>

<p><code>minimal.yaml</code> contains only system-specific or non-default values. Defaults live in <code>common/config.py</code>. The initial unbiased DCD and its matching topology are used only for iteration-0 diffusion training; no RiteWeight or Colvars trajectory is required before that training. The normal <code>Unbiased.A.conf</code>/<code>Unbiased.B.conf</code> templates provide the post-TMD trajectories consumed by the first RiteWeight step.</p>

<p><code>Workflow.iteration_noise_scales</code> overrides <code>Generative.inference.noise_scale</code> only for the listed iterations. Likewise, <code>Workflow.iteration_diffusion_epochs</code> overrides <code>Generative.training.epochs</code>; omitted iterations use the Generative fallback values. Set <code>VCN.plot_committor_projections: true</code> only when committor maps on <code>cvs_to_plot</code> should be written during <code>committor_slice</code>.</p>

<p><code>VCN.q_variance</code> sets the half-width of the committor slice around <code>q=0.5</code> and must be between 0 and 0.5. The default <code>0.1</code> selects <code>0.4 &lt;= q &lt;= 0.6</code>. <code>VCN.n_targets</code> is a simple count limit: after this filter, the workflow writes the first N candidate frames in trajectory order. It does not cluster the candidates. With <code>require_n_targets: true</code>, the step fails if fewer than N candidates are available; otherwise, it writes all available candidates.</p>

<h3>Iteration Output Layout</h3>

<pre><code>Iterations/
  0th.Iteration/
    effective_config.yaml
    workflow_manifest.json
    models/diffusion/
    generated/
    cluster_targets/
    targets/
    namd/
    riteweight/
  1st.Iteration/
    models/diffusion/
    models/vcn/
    generated/
    committor_slice/
    targets/
    namd/
    riteweight/
</code></pre>

<p><code>effective_config.yaml</code> is the fully resolved per-iteration configuration used by every stage. <code>workflow_manifest.json</code> records stage status, timestamps, errors, and attempt counts and is the source of truth for <code>--resume</code>, <code>--run_step</code>, and <code>--rerun_step</code>.</p>

<h3>Run a Specific Step</h3>

<pre><code>gen-compas --config &lt;PATH_TO_CONFIG&gt; --iteration &lt;N&gt; --run_step &lt;STEP_NAME&gt;
</code></pre>

<p>RiteWeight and FEL projection use the corresponding sections in the unified <code>config.yaml</code>:</p>

<pre><code>gen-compas --config workflow.yaml --iteration 1 --run_step riteweight
gen-compas --config workflow.yaml --iteration 1 --run_step fel_estimate
gen-compas --config workflow.yaml --iteration 1 --run_step namd
gen-compas --config workflow.yaml --iteration 1 --run_step sample_diffusion
</code></pre>

<p>VCN training and committor slicing use the same fixed-anchor internal-coordinate implementation as RiteWeight. The workflow validates the atom selection and feature settings before either VCN step runs.</p>

<p><strong>Available <code>&lt;STEP_NAME&gt;</code> options:</strong></p>
<ul>
<li><code>train_diffusion</code></li>
<li><code>sample_diffusion</code></li>
<li><code>train_committor</code></li>
<li><code>committor_slice</code></li>
<li><code>clustering</code></li>
<li><code>occupancy</code></li>
<li><code>namd</code></li>
<li><code>riteweight</code></li>
<li><code>fel_estimate</code></li>
</ul>

<p><code>clustering</code> belongs to iteration 0, while <code>train_committor</code> and <code>committor_slice</code> belong to iterations 1 and later. <code>fel_estimate</code> is available only when <code>Workflow.run_fel</code> is enabled. Invalid iteration/step combinations produce a command-line error before the stage starts.</p>

<h2 id="configuration-file-configyaml">Configuration File (config.yaml)</h2>

<p>
All parameters for model training, inference, and analysis are specified in a single YAML file.
Below is a summary of each section.
</p>

<h3>Generating a YAML File with the GUI</h3>

<p>The Gen-COMPAS configuration GUI can be used to create the YAML file for subsequent calculations. It can generate a complete configuration from scratch or load an existing minimal/complete YAML, guide the user through the supported sections, browse for input files and directories, validate essential inputs, preview the result, and save the final YAML. The saved file is then passed to the workflow with <code>gen-compas --config &lt;CONFIG.yaml&gt;</code>.</p>

<pre><code># Launch the interactive configuration helper
gen-compas-config

# Generate a complete YAML from an existing minimal configuration
gen-compas-config --config minimal.yaml --output complete.workflow.yaml

# Validate a generated or edited YAML without running calculations
gen-compas-config --config complete.workflow.yaml --validate-only
</code></pre>

<p>Tkinter is required for the interactive GUI. Lists and sparse iteration overrides should be entered using standard YAML syntax. The GUI is intended to simplify configuration generation; users should still inspect the generated YAML and perform a dry run before starting expensive calculations.</p>

<h3>Workflow Orchestration (Workflow)</h3>

<ul>
<li><strong>root_dir:</strong> Parent directory for ordinal iteration folders.</li>
<li><strong>initial_diffusion_data:</strong> DCD and matching topology used to train iteration 0 diffusion.</li>
<li><strong>initial_data_folders:</strong> Optional existing NAMD result folders included in cumulative RiteWeight input.</li>
<li><strong>run_initial_unbiased, run_fel, warm_start_diffusion, isolate_steps:</strong> Optional schedule and runtime controls.</li>
<li><strong>iteration_noise_scales, iteration_diffusion_epochs:</strong> Sparse per-iteration overrides.</li>
</ul>

<p>The workflow derives managed input/output paths for all stages and writes them to each iteration's <code>effective_config.yaml</code>. The generated <code>Workflow.runtime</code> subsection is internal and should not be added manually.</p>

<h3>1. Generative Model (Generative)</h3>

<p>Train or sample from a diffusion model that learns to generate protein structures.</p>

<p><strong>Key subsections:</strong></p>
<ul>
<li><strong>data:</strong> Input trajectory and topology paths.</li>
<li><strong>model:</strong> Embedding and architecture parameters (SchNet and attention layers).</li>
<li><strong>diffusion:</strong> Diffusion model hyperparameters (timesteps, beta schedule).</li>
<li><strong>training:</strong> Optimization and logging parameters.</li>
<li><strong>inference:</strong> Sampling configuration (checkpoint, output, batch size, etc.).</li>
</ul>

<h3>2. Variational Committor Network (VCN)</h3>

<p>Train and evaluate a committor model to predict transition probabilities between states A and B.</p>

<p><strong>Key fields:</strong></p>
<ul>
<li><strong>sampling_path, topfile, atomselect:</strong> Input trajectory data.</li>
<li><strong>epochs, learning_rate, num_layers, num_nodes:</strong> Training hyperparameters.</li>
<li><strong>gendcdfile, model_fn, slice_dir:</strong> For slicing generated trajectories by committor values.</li>
<li><strong>cvs_to_plot:</strong> For visualization (2D or 3D plots).</li>
<li><strong>plot_committor_projections:</strong> Set to <code>true</code> to write committor maps during the workflow; the default is <code>false</code>.</li>
</ul>

<h3>3. Clustering (Clustering)</h3>

<p>Cluster conformations based on atomic coordinates and extract representative structures.</p>

<p><strong>Parameters include:</strong></p>
<ul>
<li><strong>n_clusters:</strong> Number of clusters (or auto-detected if null).</li>
<li><strong>n_per_cluster:</strong> Frames per cluster to output.</li>
<li><strong>select_farthest:</strong> Include both closest and farthest structures from centroids.</li>
</ul>

<h3>4. Occupancy (Occupancy)</h3>

<p>Set atom occupancies or add hydrogens in PDB files.</p>

<p><strong>Parameters include:</strong></p>
<ul>
<li><strong>pdb_dir:</strong> Workflow-managed generated target PDBs, normally without hydrogen atoms.</li>
<li><strong>topology_file, pdb_file:</strong> Matching reference topology and coordinate PDB that include hydrogen atoms and supply the complete atom set.</li>
<li><strong>add_hydrogens:</strong> Whether to add hydrogens. <em>(Notice: Only for formatting, do NOT use hydrogens for TMD simulations)</em></li>
<li><strong>selection:</strong> MDTraj/MDAnalysis selection string for occupancy.</li>
</ul>

<h3>5. NAMD Sampling (NAMD)</h3>

<p>Copy a NAMD template directory for every target/protocol pair, replace the TMD force constant and target PDB, then run TMD followed by unbiased MD. Independent jobs can run concurrently with configurable CPU or GPU commands.</p>

<p><strong>Key options:</strong></p>
<ul>
<li><strong>namd_path, template_path:</strong> NAMD executable and reusable input directory.</li>
<li><strong>tmd_force_constant:</strong> Value inserted into the active <code>TMDk</code> directive.</li>
<li><strong>protocols:</strong> Initial-unbiased, TMD, and post-TMD unbiased templates for states A and B.</li>
<li><strong>execution:</strong> Parallel-job limit, CPU/GPU mode, device slots, threads, and command templates.</li>
</ul>

<p>TMD templates may use <code>{{TMD_FORCE_CONSTANT}}</code> and <code>{{TARGET_PDB}}</code>. The runner also replaces active <code>TMDk</code> and <code>TMDFile</code> directives directly, so the existing example templates work without conversion.</p>

<h3>6. RiteWeight (RiteWeight)</h3>

<p>
Compute statistical weights and aligned training artifacts using the
<strong>RiteWeight</strong> method described in the paper linked above.
</p>

<p><strong>Key options:</strong></p>
<ul>
<li><strong>io:</strong> Topology, output directory, and trajectory stride.</li>
<li><strong>features:</strong> Distance or internal-Z-matrix features, with optional caching.</li>
<li><strong>riteweight:</strong> Cluster count, lag, iteration, convergence, and random-seed controls.</li>
<li><strong>colvars:</strong> CV retention and optional periodic encodings.</li>
<li><strong>outputs:</strong> Torch VCN table and selected DCD/PDB files for diffusion training.</li>
</ul>

<p>RiteWeight records a numeric trajectory identifier and frame number in its VCN table. Lagged VCN samples are formed independently within each source trajectory, never across NAMD-job boundaries.</p>

<h3>7. Weighted FEL Projection (FEL_estimate)</h3>

<p>Build one or more weighted 1D/2D free-energy projections directly from the RiteWeight Torch or CSV table.</p>

<p>RiteWeight writes two distinct frame-weight columns. <code>transition_weight</code>
(also retained as the backward-compatible <code>weight</code> column) assigns each
lagged segment weight to its origin and is used by VCN training.
<code>fel_weight</code> assigns half of each segment weight to its origin and half
to its endpoint. This time-symmetric marginal includes the final lagged frames of
every trajectory and is the default for FEL projections.</p>

<p><code>FEL_estimate.landscape_F_max</code> is the high free-energy cap used
when filling unsampled bins and smoothing the landscape. A projection's
<code>F_max</code> controls only the PNG display range; the <code>.dat</code>
and <code>.npz</code> outputs retain the landscape calculated with the higher
cap.</p>

<h2>Practical Guidance for Parameter Tuning</h2>

<p>
The hyperparameters provided in the configuration file and manuscript should be treated as empirically validated settings for the systems studied here, rather than universal constants. In practice, most machine-learning hyperparameters were not tuned separately for each system. For the complex systems considered in this work, we used the same generative-model architecture, optimizer settings, training protocol, and diffusion-related hyperparameters across applications. The batch size was adjusted only when required by GPU memory limitations. The exact values used for each application are reported in the corresponding configuration files and in the manuscript.
</p>

<p>
For the Variational Committor Network (VCN), the lag time <code>tau</code> is only weakly sensitive within a broad physically meaningful range. Training is mainly affected when <code>tau</code> is chosen at pathological limits. If <code>tau</code> is too short, for example on the order of approximately 1 fs, the data may retain non-Markovian vibrational correlations. If <code>tau</code> is too long, for example on the order of approximately 1 ns, transition-region correlations may be largely lost and the training statistics can deteriorate. Between these limits, the learned committor is only weakly affected by the precise choice of <code>tau</code>. The values used in the present applications are provided in the example configuration files and in the manuscript.
</p>

<p>
The parameters that require the most practical attention are the targeted molecular dynamics (TMD) force constant and the noise level used during generative sampling. The TMD force constant should be large enough to reduce the RMSD to the generated target over the chosen TMD duration, but not so large that it produces abrupt structural distortions or unstable forces. We recommend choosing this parameter by inspecting the initial and final RMSD distributions, especially the first and final 2% of each TMD trajectory. A useful setting should produce a clear decrease in the final RMSD while maintaining structurally plausible trajectories.
</p>

<p>
For generative sampling, we recommend noise values in the approximate range <code>0</code>–<code>50</code>. Larger values can be useful during the initial iterations, especially when the available trajectories are short or sparse, because they increase the diversity of generated intermediates and help initialize subsequent sampling. After additional transition data have been accumulated, smaller values, typically <code>0</code>–<code>2</code>, are usually sufficient.
</p>

<p>
Increasing the sampling noise moves generated structures farther from the current data distribution and improves diversity, whereas decreasing the noise keeps generated structures closer to previously sampled configurations but reduces exploration. In all cases, the final acceptance criterion should not be the generative loss alone, but the downstream molecular dynamics diagnostics, including TMD convergence, bidirectional committor consistency, shooting validation where feasible, and reproducibility across independent Gen-COMPAS runs.
</p>

<h2>Dependencies</h2>

<p><strong>Core requirements:</strong></p>
<ul>
<li>Python &gt;= 3.9</li>
<li>PyTorch &gt;= 2.0</li>
<li>MDTraj, MDAnalysis</li>
<li>NumPy, SciPy, scikit-learn, pandas, matplotlib</li>
<li>PyYAML, tqdm, tensorboard</li>
</ul>

<p><strong>External simulation requirements:</strong></p>
<ul>
<li>NAMD with the bundled Colvars module. NAMD 3.0.2 or newer is recommended because 3.0.2 includes important Colvars fixes.</li>
</ul>

<h2>Gen-COMPAS Installation Guide</h2>

<p>Gen-COMPAS requires Python 3.9 or newer. A fresh environment is strongly recommended because PyTorch, MDTraj, MDAnalysis, and their compiled dependencies must be mutually compatible. NAMD and Colvars are external applications: pip does not install them, so <code>NAMD.namd_path</code> and the NAMD template files must be supplied separately.</p>

<h3>Install NAMD and Colvars</h3>

<p>Download a precompiled NAMD build for the target CPU/GPU platform from the <a href="https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD">official NAMD download page</a>, accept the NAMD license, and extract the archive. Colvars is included in NAMD and does not require a separate installation. The workflow expects templates that enable it with <code>colvars on</code> and point to a configuration file with <code>colvarsConfig</code>.</p>

<p>Either add the directory containing <code>namd3</code> to <code>PATH</code>, or configure its absolute path in the workflow YAML:</p>

<pre><code>NAMD:
  namd_path: /absolute/path/to/NAMD/namd3
  template_path: /absolute/path/to/NAMD_inputs
</code></pre>

<p>Confirm that the binary is executable before starting the workflow:</p>

<pre><code>test -x /absolute/path/to/NAMD/namd3 &amp;&amp; echo "NAMD executable found"
</code></pre>

<p>On a local multicore workstation, NAMD runs configuration files as <code>namd3 +p&lt;threads&gt; &lt;configfile&gt;</code>. Gen-COMPAS builds this command from <code>NAMD.execution</code>; cluster-specific NAMD/Charm++ launch commands can be supplied there when needed. See the <a href="https://www.ks.uiuc.edu/Research/namd/3.0.2/ug/node93.html">official NAMD workstation instructions</a> and the <a href="https://colvars.github.io/">Colvars documentation</a> for platform and configuration details.</p>

<h3>Recommended Conda Installation</h3>

<p>The repository is tested with Python 3.12. On Linux, installing the C/C++ runtime libraries from conda-forge avoids common MDAnalysis and scientific-stack loader errors:</p>

<p>If a specific CUDA-enabled PyTorch build is required, install the appropriate PyTorch build for the machine before running <code>python -m pip install .</code>; Gen-COMPAS requires <code>torch&gt;=2.0</code> but does not select a CUDA toolkit build on the user's behalf.</p>

<pre><code>conda create -n gen-compas -c conda-forge \
  python=3.12 libstdcxx-ng libgcc-ng pip -y
conda activate gen-compas

git clone https://github.com/Tangcyu/Gen-COMPAS.git
cd Gen-COMPAS
python -m pip install .
</code></pre>

<p>For development, use an editable installation so source changes are immediately visible:</p>

<pre><code>python -m pip install -e .
</code></pre>

<h3>Standard Virtual-Environment Installation</h3>

<pre><code>python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
</code></pre>

<p>Gen-COMPAS does not require <code>torch-scatter</code>; scatter-mean operations use native PyTorch.</p>

<h3>Installed Commands</h3>

<p>Installation creates <code>gen-compas</code> as the primary workflow command. <code>gen-compas-workflow</code> is an equivalent alias. The legacy top-level <code>run.py</code> interface is archived and is not installed.</p>

<pre><code>gen-compas --help
gen-compas --config /path/to/workflow.yaml --iteration 0 1 2 --dry-run
gen-compas --config /path/to/workflow.yaml --iteration 0 1 2
</code></pre>

<h3>Graphical Configuration Helper</h3>

<p>An optional graphical configuration helper is installed with Gen-COMPAS and can be used to generate YAML files for later workflow calculations. See <a href="#configuration-file-configyaml">Configuration File (config.yaml)</a> for its usage and validation options.</p>

<pre><code>gen-compas-config
</code></pre>

<h3>Verify the Installation</h3>

<pre><code>python -c "import torch, mdtraj, MDAnalysis; print('scientific stack ok')"
python -c "import workflow, common.runner; print('Gen-COMPAS import ok')"
gen-compas --help
</code></pre>

<p>The help output should list <code>--run_step</code>, <code>--rerun_step</code>, all valid step names, and both iteration schedules.</p>

<h3>Run Directly from a Source Checkout</h3>

<p>Installation is recommended, but the workflow can be executed directly with the same environment. The path to <code>workflow.py</code> may be relative or absolute, and the command may be launched from a separate simulation directory:</p>

<pre><code>python /path/to/Gen-COMPAS/workflow.py \
  --config ./trpcage.workflow.yaml \
  --iteration 0 \
  --stepwise
</code></pre>

<p>Isolated stages automatically make the sibling Gen-COMPAS packages discoverable without changing the caller's working directory.</p>

<h3>Troubleshooting and Wheel Installation</h3>

<p>If an existing conda environment reports a <code>CXXABI</code> or <code>libstdc++.so.6</code> error, update the conda-forge runtime libraries:</p>

<pre><code>conda install -n gen-compas -c conda-forge libstdcxx-ng libgcc-ng -y
</code></pre>

<p>To build and install a wheel locally:</p>

<pre><code>python -m pip wheel --no-deps . -w dist
python -m pip install dist/Gen_COMPAS-*.whl
</code></pre>

<hr/>

<h2>Demo</h2>

<p>The initial training data for the Trp-cage fast-folding protein is located at:</p>

<pre><code>example/0.DEMO_Trp-cage/Dataset/
</code></pre>

<p>On an NVIDIA L40s GPU:</p>

<ul>
<li>Training the diffusion model for 50 epochs takes approximately <strong>5 minutes</strong>, producing PyTorch checkpoints (.pt).</li>
<li>Generating 1,000 structures (.pdb format) using batch size 200 takes approximately <strong>1 minute</strong>.</li>
</ul>

<hr/>

<h2>Reproducibility</h2>

<p>
The <code>example/</code> folder contains all molecular dynamics input files used in the paper, prepared for the NAMD and Colvars software packages. These include:
</p>

<ul>
<li>Topologies, coordinates, velocities, and periodic boxes</li>
<li>Force field parameter files</li>
<li>Systems for NANMA, Tri-alanine, Trp-cage, RBP, and AAC</li>
</ul>

<p>
Running the complete Gen-COMPAS workflow on these inputs will reproduce the results presented in the manuscript.
</p>

</body>
</html>
