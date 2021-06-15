# STARMap spatial transcriptome analysis toolkit

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/jiahaoh/starmap_clean/graphs/commit-activity)

The python package of STARMap bioinformatics analysis pipeline.

## Installation
```python
# via Pypi
pip install starmap
```

## Usage
```python
# Basic 
from starmap.obj import STARMapDataset, load_data
import starmap.analyze as anz
import starmap.viz as viz
```

## Workflow
![pipeline](https://jiahaoh.com/project/pipeline_example.png)
cited from Wang(2018),
more details @ [example](https://github.com/jiahaoh/starmap_clean/tree/master/examples)

## TODO
- Clustering algorithm reproducibility
- Include more example jupyter notebooks
- Conversion support of Scanpy, Seurat 
- Batch effect module
- Detailed function documentation

## Change Log
- v0.0.1 - Package creation

## Reference 
1. X Wang*, W E Allen*, M Wright, E Sylwestrak, N Samusik, S Vesuna, K Evans, C Liu, C Ramakrishnan, J Liu, G P Nolan#, F-A Bava#, K Deisseroth#. Three-dimensional intact-tissue-sequencing of single-cell transcriptional states. Science 2018, eaat5691.

_*co-first authors; #corresponding authors_

## License
[MIT License](https://github.com/jiahaoh/starmap_clean/blob/master/LICENSE.txt)