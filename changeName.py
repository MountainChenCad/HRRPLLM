import os


def rename_mat_files_in_directory(directory_path="."):
    """
    Renames .mat files in a specified directory based on a predefined mapping.
    The naming format is assumed to be A_B_C_D.mat, where 'A' is the target for renaming.
    It also handles cases like A.mat or A_X.mat.

    Args:
        directory_path (str): The path to the directory containing .mat files.
                              Defaults to the current directory (".").
    """
    name_map = {
        "幻影2000": "Mirage2000",
        "全球鹰": "GlobalHawk",
        "捕食者": "Predator"
    }

    renamed_count = 0
    processed_count = 0

    # Verify the directory exists
    if not os.path.isdir(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在。")
        return

    print(f"正在扫描目录: {os.path.abspath(directory_path)}")

    for filename in os.listdir(directory_path):
        if filename.endswith(".mat"):
            processed_count += 1
            # Remove the .mat extension to work with the base name
            base_name = filename[:-4]  # Removes last 4 characters (".mat")

            # Split the base name by the first underscore to get the 'A' part
            # If no underscore, 'type_a' will be the whole base_name
            if '_' in base_name:
                type_a = base_name.split('_', 1)[0]
                rest_of_name = base_name.split('_', 1)[1]
            else:
                type_a = base_name
                rest_of_name = None  # No further parts after A

            if type_a in name_map:
                new_type_a = name_map[type_a]

                # Reconstruct the new base name
                if rest_of_name is not None:
                    new_base_name = f"{new_type_a}_{rest_of_name}"
                else:
                    new_base_name = new_type_a

                new_filename = new_base_name + ".mat"

                old_filepath = os.path.join(directory_path, filename)
                new_filepath = os.path.join(directory_path, new_filename)

                if old_filepath == new_filepath:
                    print(f"文件 '{filename}' 无需改名 (可能已是目标格式或首部不在映射中)。")
                    continue

                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"已重命名: '{filename}' -> '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"重命名文件 '{filename}' 失败: {e}")
            # else:
            #     print(f"跳过 '{filename}': 类型 '{type_a}' 不在映射中或已经是英文。")

    print("\n--- 报告 ---")
    print(f"共处理了 {processed_count} 个 .mat 文件。")
    print(f"成功重命名了 {renamed_count} 个文件。")
    if processed_count == 0:
        print("在指定目录中没有找到 .mat 文件。")
    elif renamed_count == 0 and processed_count > 0:
        print("没有文件符合重命名条件，或它们已经是目标名称。")
    print("脚本执行完毕。")


if __name__ == "__main__":
    # #########################################################################
    # # 重要：请将下面的 "your_mat_files_directory_path"             #
    # # 替换为您的 .mat 文件所在的实际文件夹路径。                         #
    # # 例如: "/path/to/your/mat_files" (Linux/macOS)                 #
    # # 或者: "C:\\path\\to\\your\\mat_files" (Windows)               #
    # # 如果脚本与 .mat 文件在同一目录下，可以使用 "."                     #
    # #########################################################################

    target_directory = "datasets/simulated_hrrp"  # 默认设置为当前目录

    rename_mat_files_in_directory(target_directory)
