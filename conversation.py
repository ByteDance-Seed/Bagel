model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    os.makedirs(args.out_dir, exist_ok=True)
    prompt = 'Answer the question using a single word or phrase.'

    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(args.out_dir, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt
            img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, img)
            if not os.path.exists(img_path):
                img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, "images", img)
            if not os.path.exists(img_path):
                continue
            images = [Image.open(img_path).convert('RGB')]
            images, conversation = process_conversation(images, question)

            response = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images,
                prompt=conversation,
                max_length=20,
            )
            response = post_processing(response)
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()