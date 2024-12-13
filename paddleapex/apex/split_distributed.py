import json

def split_json_by_keyword(input_file, outfiles, keywords):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 分别存储包含和不包含关键字的项
    out_list = []
    for i in range(len(keywords)):
        out_data = {}
        out_list.append(out_data)

    without_keyword = {}
    
    # 遍历每个项并分类
    for key, value in data.items():
        have_key = False
        for keyword, out_data in zip(keywords, out_list):
            if keyword in key:
                out_data[key] = value
                have_key = True
        if not have_key:
            without_keyword[key]= value
    
    for i in range(len(keywords)):
        output_file_with = outfiles[i]
        with_keyword = out_list[i]
        with open(output_file_with, 'w', encoding='utf-8') as f_with:
            json.dump(with_keyword, f_with, ensure_ascii=False, indent=4)
    
    output_file_without = outfiles[-1]
    with open(output_file_without, 'w', encoding='utf-8') as f_without:
        json.dump(without_keyword, f_without, ensure_ascii=False, indent=4)

    print("well done")

input_json_files =       ["/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/forward_rank0_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/forward_rank1_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/forward_rank2_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/forward_rank3_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/forward_rank4_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/forward_rank5_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/forward_rank6_all.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/forward_rank7_all.json"]
distributed_keyword    = ["/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/distributed.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/distributed.json"]
model_keyword          = ["/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/class.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/class.json"]
common_keyword        =  ["/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/common.json",
                          "/ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/common.json"]



keyword = ['distributed', 'model']
for i in range(len(input_json_files)):
    split_json_by_keyword(input_json_files[i], [distributed_keyword[i], model_keyword[i], common_keyword[i]],keyword)

