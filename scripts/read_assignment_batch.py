import sys
from starmap.sequencing import *


def main():
    # Input
    base_path = sys.argv[1]
    reads_file = sys.argv[2]
    run_id = sys.argv[3]
    assign_reads = True

    # Get folder for each position
    data_folders = sorted([f for f in os.listdir(base_path) if f.startswith('position')])
    # data_folders = data_folders[0:1] # test

    # Load gene information (genes.csv)
    genes2seq, seq2genes = load_genes(base_path)

    # Iterate through each folder
    for i, folder in enumerate(data_folders):
        curr_path = os.path.join(base_path, folder)

        print(f"====Processing: {folder}====")
        out_path = os.path.join(curr_path, 'output')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # Load reads information (read_file)
        bases, points = load_read_position(curr_path, reads_file)
        print(f"Number of reads: {len(bases)}")

        temp_path = f"../new_seg/{run_id}"
        label_path = os.path.join(temp_path, folder)

        labels = load_label_image(label_path)
        plot_cell_numbers_3d(out_path, labels)

        # Plot reads on Nissl
        seg = load_seg_image(label_path)
        plt.figure(figsize=(80, 40))
        plt.plot(points[:, 1], points[:, 0], 'r.', markersize=4)
        plt.imshow(seg, cmap=plt.cm.get_cmap('gray'))
        plt.axis('off')
        points_seg_path = os.path.join(out_path, "points_seg.png")
        print("Saving points_seg.png")
        plt.savefig(points_seg_path)

        if assign_reads:
            reads_label, Nlabels = assign_reads_to_cells_3d(labels, points)
            convert_reads_assignment_3d(out_path, run_id, reads_label, Nlabels, bases, seq2genes)

        print("====Finished====")


if __name__ == "__main__":
    main()
