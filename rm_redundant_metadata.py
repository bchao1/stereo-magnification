import os

if __name__ == "__main__":
    metadata_dir = "/mnt/MPI/camera_metadata/train"
    img_dir = "/mnt/MPI/images/train"
    metadata = sorted(os.listdir(metadata_dir))
    images = sorted(os.listdir(img_dir))

    need2delete = []
    donotrm = []

    for file in metadata:
        file_dir = os.path.join(metadata_dir, file)
        with open(file_dir, "r") as txtfile:
            url = txtfile.readline()
            file_id = url[-12 : -1]
            if file_id not in images:
                need2delete.append(file)
            else:
                donotrm.append((file, file_id))

    for d in need2delete:
        redundant_dir = os.path.join(metadata_dir, d)
        print("Removing %s" % redundant_dir)
        os.remove(redundant_dir)

    # donotrm = sorted(donotrm, key=lambda x: x[1])
    # print(len(donotrm))
    # for d in donotrm:
    #     print(d)
