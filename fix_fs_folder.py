from os.path import join as pj
import os


def find_reference(ref_folder, filename):
    def __match(ref, query):
        # ref: _C_xxx.bvh
        # query: STY_xxx.bvh
        r1 = ref[3:] == query[4:] or ref[3:] == query[3:]
        r2 = ref == query[-len(ref):]
        print(query[-len(ref):])
        return r1 or r2

    for root, dirs, filenames in os.walk(ref_folder):
        for nm in filenames:
            if __match(nm, filename):
                return pj(root, nm)


def remove_fs_folder(inp_folder, ref_folder, dst_folder):
    from remove_fs import remove_fs
    os.makedirs(dst_folder, exist_ok=True)

    for root, dirs, filenames in os.walk(inp_folder):
        for filename in filenames:
            dst_file = pj(dst_folder, filename)
            if os.path.isfile(dst_file):
                continue

            ref_file = find_reference(ref_folder, filename)
            if ref_file is None:
                print(f"warning: no matching result for {filename}")
                continue
            remove_fs(pj(root, filename), ref_file, dst_file, 5, 13, True)

