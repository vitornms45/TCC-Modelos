import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carregamento dos Dados DIRETAMENTE do arquivo Excel ---
# O parâmetro 'sheet_name' permite especificar a aba desejada.
file_path = 'Resultados_100_Rodadas_Keras_YOLO_com_Recall_Precisao.xlsx'
try:
    # Tenta ler do arquivo Excel. Requer a biblioteca 'openpyxl'.
    df_keras = pd.read_excel(file_path, sheet_name='Keras')
    df_yolo = pd.read_excel(file_path, sheet_name='YOLOv11')
    
    # Adiciona a coluna 'Modelo' para identificação
    df_keras['Modelo'] = 'Keras CNN'
    df_yolo['Modelo'] = 'YOLOv11'
    
    # Combina os DataFrames
    df_combined = pd.concat([df_keras, df_yolo], ignore_index=True)
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    print("Certifique-se de que ele está no mesmo diretório do script.")
    exit()
except Exception as e:
    # Se o erro for outro (ex: 'openpyxl' não instalado), avisa o usuário.
    print(f"Ocorreu um erro ao ler o arquivo Excel: {e}")
    print("Verifique se a biblioteca 'openpyxl' está instalada (pip install openpyxl).")
    exit()


# --- 2. Geração de Gráficos Individuais ---

# Define um tema visual e uma paleta de cores
sns.set_theme(style="whitegrid", font_scale=1.1, rc={"axes.facecolor": "#f0f0f0", "figure.facecolor": "white"})
palette = sns.color_palette("viridis", 2)

metrics_and_time = ['Acurácia', 'Precisão', 'Recall', 'F1-score', 'AUC', 'Tempo Inferência (s)']

# Loop para criar uma imagem separada para cada métrica
for metric in metrics_and_time:
    # Cria uma nova figura para cada gráfico
    plt.figure(figsize=(10, 7))
    ax = plt.gca() # Pega o eixo atual para plotagem

    # Plota o gráfico de densidade com contorno
    sns.kdeplot(data=df_combined, x=metric, hue='Modelo', 
                fill=True, alpha=0.5, palette=palette, 
                linewidth=2, ax=ax)
    
    # Adiciona um "tapete" (rugplot) na base do gráfico
    sns.rugplot(data=df_combined, x=metric, hue='Modelo', palette=palette, ax=ax, height=0.05, legend=False)
    
    # Calcula e anota as médias
    mean_keras = df_combined[df_combined['Modelo'] == 'Keras CNN'][metric].mean()
    mean_yolo = df_combined[df_combined['Modelo'] == 'YOLOv11'][metric].mean()
    
    ax.axvline(mean_keras, color=palette[0], linestyle='--', linewidth=2.5)
    ax.axvline(mean_yolo, color=palette[1], linestyle='--', linewidth=2.5)
    
    y_pos = ax.get_ylim()[1] * 0.9
    
    ax.text(mean_keras, y_pos, f'{mean_keras:.4f}', 
            ha='center', va='center', color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=palette[0], ec='black', lw=1))
    
    ax.text(mean_yolo, y_pos, f'{mean_yolo:.4f}', 
            ha='center', va='center', color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=palette[1], ec='black', lw=1))

    # Define os títulos e legendas para o gráfico individual
    ax.set_title(f'Análise de Densidade - {metric}', fontsize=18, weight='bold')
    ax.set_xlabel(f'Valor da {metric}', fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
   

    # Cria um nome de arquivo padronizado
    clean_metric_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    file_name = f'grafico_densidade_{clean_metric_name}.png'
    
    # Salva a figura e a fecha para a próxima iteração
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close() # Fecha a figura para liberar memória
    
    print(f"Gráfico para '{metric}' salvo como '{file_name}'")