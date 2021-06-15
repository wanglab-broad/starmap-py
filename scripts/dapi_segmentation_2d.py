import sys
from starmap.sequencing import *


def main():
    # Input
    base_path = sys.argv[1]

    # Start processing
    print(f"====Processing: {base_path}====")
    out_path = os.path.join(base_path, 'output')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Load manual markers (CellCounter.xml)
    cell_locs = parse_CellCounter(os.path.join(base_path, "CellCounter.xml"))

    # Load 2D DAPI image (nissl_maxproj_resized.tif)
    dapi = load_nissl_image(base_path, "dapi_maxproj_resized.tif")

    # Run segmentation
    labels = segment_nissl_image(base_path, dapi, cell_locs.astype(np.int), dilation=False)
    plot_cell_numbers(base_path, labels)
    plt.imsave(fname=os.path.join(out_path, "dapi_labels.tiff"), arr=labels)

    print("====Finished====")


if __name__ == "__main__":
    main()

