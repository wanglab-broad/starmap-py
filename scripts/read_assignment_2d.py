import sys
from starmap.sequencing import *


def main():
    # Input
    base_path = sys.argv[1]
    reads_file = sys.argv[2]
    run_id = sys.argv[3]
    assign_reads = True

    # Start processing
    print(f"====Processing: {base_path}====")
    out_path = os.path.join(base_path, 'output')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Load gene information (genes.csv)
    genes2seq, seq2genes = load_genes(base_path)

    # Load reads information (read_file)
    bases, points = load_read_position(base_path, reads_file)
    print(f"Number of reads: {len(bases)}")

    points[:, 0] = points[:, 0] + 120
    points[:, 1] = points[:, 1] + 20

    # Load manual markers (CellCounter.xml)
    cell_locs = parse_CellCounter(os.path.join(base_path, "CellCounter.xml"))

    # Load 2D Nissl image (nissl_maxproj_resized.tif)
    nissl = load_nissl_image(base_path, "nissl_maxproj_resized.tif")

    # Run segmentation
    labels = segment_nissl_image(base_path, nissl, cell_locs.astype(np.int))
    plot_cell_numbers(base_path, labels)
    plt.imsave(fname=os.path.join(out_path, "labels.tiff"), arr=labels)

    # labels = load_label_image(base_path, fname='stardist.tif')

    # Plot reads on Nissl
    plt.figure(figsize=(80, 40))
    plt.plot(points[:, 1], points[:, 0], 'r.', markersize=0.5)
    plt.imshow(labels>0, cmap=plt.cm.get_cmap('gray'))
    plt.axis('off')
    points_seg_path = os.path.join(out_path, "points_seg.png")
    print(f"Saving points_seg.png")
    plt.savefig(points_seg_path)

    # Reads assignment
    if assign_reads:
        hulls, point_assignments, coords = assign_reads_to_qhulls(labels, points)
        convert_reads_assignment_qhull(base_path, run_id, point_assignments, bases, seq2genes)
        print("Saving labels.npz")
        np.savez(os.path.join(out_path, "labels.npz"), labels=labels)

    print("====Finished====")


if __name__ == "__main__":
    main()
