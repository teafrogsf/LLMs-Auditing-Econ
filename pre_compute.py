import yaml

model_infos = yaml.safe_load(open('config/toy_game/model_config.yaml'))
print(model_infos)
L = 1500


def comput_provider_info(provider_idx, L=L):
    minfo = [model_infos[f'provider-{provider_idx}-good'], model_infos[f'provider-{provider_idx}-mid'],  model_infos[f'provider-{provider_idx}-weak']]

    

    max_score_mu = minfo[0]['score_mu'] 
    max_token_mu = minfo[0]['output_tokens_mu']

    gamma = minfo[0]['output_tokens_mu']/L
    print(f'for provider {provider_idx}, gamma = {gamma}')
    for i in range(2):
        
        p_i = minfo[0]['output_token_price']
        h_x_grad = minfo[i]['score_mu'] / max_score_mu - minfo[i+1]['score_mu'] / max_score_mu

        h_x_grad /= float(minfo[i]['output_token_price']) - float(minfo[i+1]['output_token_price'])

        h_x = minfo[i+1]['score_mu'] / max_score_mu

        
    

        g_x_grad = minfo[i]['output_tokens_mu'] / max_token_mu - minfo[i+1]['output_tokens_mu'] / max_token_mu

        g_x = minfo[i+1]['score_mu'] / max_token_mu


        ss =  (h_x_grad / h_x - g_x_grad / g_x) * (p_i*0.9)

        print(f'for provider {provider_idx}, model {i}: ss = {ss}')




if __name__ == '__main__':
    for i in range(1, 4):
        comput_provider_info(i)