# Importar as bibliotecas 
# Precisamos do streamlit, numpu, PIL e o pytorch
# Aqui precisamos só carregar o modelo já treinado!
import streamlit as st
import numpy as np
from PIL import Image
import time

import torch
from torchvision import transforms

# definindo as transformações para imagens novas a serem submetidas ao modelo!
image_size = 100

# Transformando as imagens
redimensionamento_imagem = transforms.Compose([
        transforms.Resize(size=[image_size, image_size]),
        transforms.ToTensor(),
    ])

def cachorro_ou_gato(model, test_image):
    '''
    Função para realizar a predição do status do AR
    Parâmetros
        :param model: modelo para testar
        :param test_image_name: imagem teste
    '''
    transform = redimensionamento_imagem

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)

    # Não precisa atualizar os coeficientes do modelo
    with torch.no_grad():
        model.eval()

        # Modelo retorna as probabilidades em log (log softmax)
        out = model(test_image_tensor)

        # torch.exp para voltar a probabilidade de log para a probabilidade linear
        ps = torch.exp(out)

        # topk retorna o os k maiores valores do tensor
        # o tensor de probabilidades vai trazer na 1a posição a classe com maior
        # probabilidade de predição
        topk, topclass = ps.topk(2, dim=1)



        classe_com_maior_prob = np.argmax(topk.cpu().numpy()[0])

    return topclass[0][0]


# Designing the interface
st.title("É Cachorro ou gato?")
# For newline
st.write('\n')

image = Image.open('bug.jpg')
show = st.image(image, use_column_width=True)

st.sidebar.title("Suba uma imagem para teste!")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Imagem enviada', use_column_width=True)
    # We preprocess the image to fit in algorithm.

# For newline
st.sidebar.write('\n')

# Carregar o modelo

modelo = torch.load('./modelos/melhor_modelo.pt')

if st.sidebar.button("Clique aqui para saber sé cachorro ou gato"):
    if uploaded_file is None:

        st.sidebar.write("Suba uma imagem")

    else:

        with st.spinner('Verificando'):

            prediction = cachorro_ou_gato(modelo, u_img)
            time.sleep(2)
            st.success('Pronto!')

        st.sidebar.header("Bug disse que a imagem é...")

        print(prediction)

        if prediction == 0:
            st.sidebar.write("É um cachorro!", '\n')
            #show.image('./images/tim_maia_feliz.png', 'Tim Maia tá felizão!', use_column_width=True)
        elif prediction == 1:
            st.sidebar.write("É um gato!", '\n')
           