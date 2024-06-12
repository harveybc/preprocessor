# Función para limpiar y normalizar el texto
def clean_text(text):
    import re
    # Eliminar caracteres especiales y normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Ruta al archivo PDF
pdf_path = 'articulo.txt'

# Extraer y limpiar el texto
f = open(pdf_path, 'r')
raw_text = f.read()
f.close()

cleaned_text = clean_text(raw_text)

# Guardar el texto limpio en un archivo para su posterior procesamiento
with open('articulo_limpio.txt', 'w', encoding='utf-8') as file:
    file.write(cleaned_text)
