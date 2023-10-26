import preparar_dataset
import preprocessar
import dividir_dataset
import criar_modelo
import inferencia

def main():
    preparar_dataset.preparar_dataset()
    preprocessar.pre_processar()
    dividir_dataset.dividir_dataset()
    criar_modelo.criar_e_treinar_modelo()
    inferencia.inferencia_imagem(r'C:\Users\gabsp\OneDrive\Documentos\mascara_view\rosto.jpg', r'C:\Users\gabsp\OneDrive\Documentos\mascara_view\model_epoch_49.pt')

if __name__ == "__main__":
    main()
