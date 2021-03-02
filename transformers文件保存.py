from transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = "./models/"

# 步骤1：保存一个经过微调的模型、配置和词汇表

#如果我们有一个分布式模型，只保存封装的模型
#它包装在PyTorch DistributedDataParallel或DataParallel中
model_to_save = model.module if hasattr(model, 'module') else model
#如果使用预定义的名称保存，则可以使用`from_pretrained`加载
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)

# 步骤2: 重新加载保存的模型

#Bert模型示例
model = BertForQuestionAnswering.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)  # Add specific options if needed
#GPT模型示例
model = OpenAIGPTDoubleHeadsModel.from_pretrained(output_dir)
tokenizer = OpenAIGPTTokenizer.from_pretrained(output_dir)

# 直接搞BERT
modelConfig = BertConfig.from_pretrained('bert-base-uncased-config.json')
self.textExtractor = BertModel.from_pretrained(
            'bert-base-uncased-pytorch_model.bin', config=modelConfig)

