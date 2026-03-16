# DETECTAR-CAM  
  
- No codigo tem as linhas:  
3 # Mapeamento de IDs para nomes.  
4 nomes = {  
5   1: "Yuri",  
6   2: "Isa",  
7 }  
  
Lista de nomes por ID  
  
- No codigo tem as linhas:  
11 classificarFace = cv2.CascadeClassifier(  
12     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  
13 )  
14   
15 identificador = cv2.face.LBPHFaceRecognizer_create()  
16 identificador.read('classificadorTreinado.yml')  
..  
28 id, acuracia = identificador.predict(imagemFace)  
  
classificarFace usa um modulo de reconhecimento do cv2  
O identificador é um arquivo yml de reconhecimento facial  
Na linha 28 ele reconhece a pessoa com base nesse arquivo  
  
- No codigo tem as linhas:  
34 id, acuracia = identificador.predict(imagemFace)  
35  
36 # No LBPH, quanto menor a acuracia, melhor o reconhecimento.  
37 if acuracia <= 55:  
38   nome = nomes.get(id, f"ID {id}")  
39   texto = f"{nome} ({acuracia:.1f})"  
40   cor = (0, 255, 0)  
41 else:  
42   texto = f"Desconhecido ({acuracia:.1f})"  
43   cor = (0, 0, 255)  
..  
55 cv2.putText(frame, texto, (x - 7, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
  
A acuracia é quanto o codigo acredita na previsão, quanto menor o numero mais chance de ser a pessoa  
Na linha 38 ele vai pegar o nome da lista 'nomes' e vai deixar a acuracia com um decimal   
se a acuracia for maior que 55 a pessoa vai ser informada como 'Desconhecido"  
  
- No codigo tem as linhas:  
45 contador = str(detectaFace.shape[0])  
..  
57 cv2.putText(frame, 'Quantidade de faces detectadas:' + contador, (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)  
  
a quantidade de pessoas é contadas pelo contador  
na linha 57 ele printa esse valor a baixo na tela  