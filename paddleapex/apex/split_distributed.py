import json

def split_json_by_keyword(input_file, output_file_with, output_file_without, keyword):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 分别存储包含和不包含关键字的项
    with_keyword = {}
    without_keyword = {}

    # 遍历每个项并分类
    for key, value in data.items():
        if keyword in key:
            with_keyword[key] = value
        else:
            without_keyword[key]= value

    # 将结果写入不同的文件
    with open(output_file_with, 'w', encoding='utf-8') as f_with:
        json.dump(with_keyword, f_with, ensure_ascii=False, indent=4)

    with open(output_file_without, 'w', encoding='utf-8') as f_without:
        json.dump(without_keyword, f_without, ensure_ascii=False, indent=4)

    print(f"Items with '{keyword}' written to {output_file_with}")
    print(f"Items without '{keyword}' written to {output_file_without}")


input_json_files = ["/zhouxiangquan/llama10b/dump_info/rank0_step5/forward_rank0_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank1_step5/forward_rank1_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank2_step5/forward_rank2_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank3_step5/forward_rank3_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank4_step5/forward_rank4_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank5_step5/forward_rank5_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank6_step5/forward_rank6_all.json",
                    "/zhouxiangquan/llama10b/dump_info/rank7_step5/forward_rank7_all.json"]

output_with_keyword = ["llama10b/rand0_distributed.json",
                       "llama10b/rand1_distributed.json",
                       "llama10b/rand2_distributed.json",
                       "llama10b/rand3_distributed.json",
                       "llama10b/rand4_distributed.json",
                       "llama10b/rand5_distributed.json",
                       "llama10b/rand6_distributed.json",
                       "llama10b/rand7_distributed.json"]

output_without_keyword = ["llama10b/rand0_without_distributed.json",
                          "llama10b/rand1_without_distributed.json",
                          "llama10b/rand2_without_distributed.json",
                          "llama10b/rand3_without_distributed.json",
                          "llama10b/rand4_without_distributed.json",
                          "llama10b/rand5_without_distributed.json",
                          "llama10b/rand6_without_distributed.json",
                          "llama10b/rand7_without_distributed.json"]

keyword = 'distributed'
for i in range(len(input_json_files)):
    split_json_by_keyword(input_json_files[i], output_with_keyword[i], output_without_keyword[i], keyword)

