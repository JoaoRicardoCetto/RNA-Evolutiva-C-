/*
  
  Rede Neural Artificial Evolutiva (RNA-E)
  
  Os pesos s�o atualizados a partir de um algoritmo
  gen�tico que busca minimizar os erros na fase de
  treinamento.
  
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_Entradas 2
#define MAX_Pesos 6

//===| Estrutura de Dados |==========================
typedef char string[60];

typedef struct tipoLicao {
	int p;  //proposi��o P
	int q;	//Proposi��o Q
	int resultadoEsperado; //Proposi��o Composta P "E" Q (A Classe)
	tipoLicao *prox;
}TLicao;

typedef struct tipoIndividuo {
	float genes[MAX_Pesos];
	int erros;
	int numero; //numero identificador
	tipoIndividuo  *prox;
}TIndividuo;

typedef struct tipoSinapse {
	int camada;
	int neuronio_origem;
	int neuronio_destino;
	float peso;
	tipoSinapse *prox;
}TSinapse;

typedef struct tipoNeuronio {
	int neuronio;
	float soma;
	float peso;
	tipoNeuronio *prox;
}TNeuronio;

typedef struct tipoLista{
	FILE *fp; //Arquivo de Sa�da (Relat�rio).
	string objetivo;
	TLicao *licoes; //Conjunto de li��es a serem aprendidas
	float entradas[MAX_Entradas];
	TNeuronio *neuronios;
	TSinapse *pesos;
	TIndividuo *populacao;
	TIndividuo *individuoAtual;
	int totalIndividuos;
	int Qtd_Populacao;
	int Qtd_Mutacoes_por_vez;
	int Total_geracoes;
	int geracao_atual;
	int Qtd_Geracoes_para_Mutacoes; 
	float sinapseThreshold;
	float learningRate;
}TLista;

TLista lista;

//====| Assinatura de Fun��es |=======================
void inicializa(TLista *L);
void geraIndividuos(TLista *L);
void geraLicoes(TLista *L);
void insereLicao(TLista *L, int p, int q, int resultado);
void insereNeuronio(TLista *L, int neuronio);
void estabelecendoSinapse(TLista *L,int neuronioDe, int neuronioAte, int camada);
void treinamento(TLista *L);
void cruzamento(TLista *L, int flag);
void avaliacaoIndividuos(TLista *L);
void ordenamentoIndividuos(TLista *L);
void promoveMutacoes(TLista *L);
void poda(TLista *L);

//===| Fun��es |======================================
void inicializa(TLista *L){
	int i;
	
	L->licoes = NULL;
	L->populacao = NULL;
	L->individuoAtual = NULL;
	L->totalIndividuos = 0;
	
	for(i=0; i < MAX_Entradas; i++){
		L->entradas[i] = 0;
	}//for
	
	L->neuronios = NULL;
	L->pesos = NULL;
	
	printf("\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====");
	printf("\n\n\t\t=====| Configuracao da RNA |=====\n\n");
	printf("\tInforme o TAMANHO da POPULACAO (em termos de individuos):\n");
	printf("\t\tSugestao: 300 individuos.\n\t\tValor: ");
	scanf("%d", &L->Qtd_Populacao);
	
	geraIndividuos(L);
	
	printf("\n\n\tInforme a QUANTIDADE de GERACOES maxima:");
	printf("\n\tSugestao: 100 geracoes no total.\n\tValor: ");
	scanf("%d", &L->Total_geracoes);
	
	L->geracao_atual = 0;
	
	printf("\n\n\tInforme o INTERVALO de GERACOES para a ocorrencia de MUTACOES:");
	printf("\n\tSugestao: 5 (a cada 5 geracoes devem ocorrer mutacoes).\n\tValor: ");
	scanf("%d", &L->Qtd_Geracoes_para_Mutacoes);
	
	printf("\n\n\tInforme a QUANTIDADE de MUTACOES que devem ocorrer POR VEZ:");
	printf("\n\tSugestao: 3 mutacoes por intervalo.\n\tValor: ");
	scanf("%d", &L->Qtd_Mutacoes_por_vez);
	
	printf("\n\nSINAPSE THRESHOLD (Limiar das Conexoes entre Neuronios):\n");
	printf("Define a intensidade do sinal que sensibiliza cada neuronio.\n\n");
	printf("\tInforme o SINAPSE THRESHOLD:\n\tSugestao: 0.60\n\tValor: ");
	scanf("%f",&L->sinapseThreshold);
	
	printf("\n\nLEARNING RATE (Taxa de Aprendizado): variacao dos pesos em cada ajuste (Aprendizado).\n");
	printf("\n\tLEARNING RATE:\n\tSugestao: 0.20\n\tValor: ");
	scanf("%f",&L->learningRate);
	
	strcpy(L->objetivo,"Aprendizado da Funcao Logica P E Q");
	
	printf("\n\n\tDefinindo as LICOES a serem aprendidas pela Rede Neural Artificial.\n\n");
	geraLicoes(L);
	
	printf("\n\n\tDefinindo os NEURONIOS que compoem a REDE NEURAL ARTIFICIAL.");
	insereNeuronio(L,1);
	insereNeuronio(L, 2);
	insereNeuronio(L, 3);
	insereNeuronio(L, 4);
	insereNeuronio(L, 5);
	
	printf("\n\n\tEstabelecendo as CONEXOES (Sinapses) entre os NEURONIOS.");
	estabelecendoSinapse(L,1,3,0);
	estabelecendoSinapse(L,1,4,0);
	estabelecendoSinapse(L,2,3,0);
	estabelecendoSinapse(L,2,4,0);
	estabelecendoSinapse(L,3,5, 1);
	estabelecendoSinapse(L,4,5, 1);
	
	L->fp = fopen("RNA_EVOLUTIVA_RELATORIO.TXT","w");
	
	fprintf(L->fp,"\n\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====\n\n");
	fprintf(L->fp,"\tOBJETIVO: %s.\n\n\tLicoes:\n", L->objetivo);
	fprintf(L->fp,"\t LICAO    P    Q  (Resultado Esperado)\n");
	fprintf(L->fp,"\t+------+----+----+---------------------+\n");
	
	TLicao *licao = L->licoes;
	int cont = 0;
	while(licao != NULL){
		fprintf(L->fp,"\t(%d) - %d   %d   %d\n", ++cont, licao->p, licao->q, licao->resultadoEsperado);
		licao = licao->prox;
	}//while
	
	fprintf(L->fp,"\n\n");
	fprintf(L->fp,"\tLearning Rate: %.2f\n", L->learningRate);
	fprintf(L->fp,"\tSinapse Threshold: %.2f\n", L->sinapseThreshold);
	fprintf(L->fp,"\tPopulacao MAXIMA: %d.\n", L->Qtd_Populacao);
	fprintf(L->fp,"\t%d MUTACOES a cada sequencia de %d GERACOES.\n", L->Qtd_Mutacoes_por_vez, L->Qtd_Geracoes_para_Mutacoes);
	fprintf(L->fp,"\tTOTAL de GERACOES: %d.\n\n\n", L->Total_geracoes);
	
	printf("\n\n\tConfiguracao FINALIZADA!!!\n\n");
	
}
//====================================================
void geraIndividuos(TLista *L){
	TIndividuo *novo;
	int i, x;
	
    srand( (unsigned)time(NULL) );
    
    for(i= 0; i < L->Qtd_Populacao; i++){
    	novo = (TIndividuo *)malloc(sizeof(TIndividuo));
		
		novo->prox = NULL;
		novo->numero = i+1;		
		novo->erros = -1;
		
		for(x=0; x < MAX_Pesos; x++){
			novo->genes[x] = rand() % 101;
			novo->genes[x] = novo->genes[x] / 100;
		}//for
		
		if(L->populacao == NULL){
			L->populacao = novo;
		} else {
			TIndividuo *atual = L->populacao;
			
			while(atual->prox != NULL){
				atual = atual->prox;
			}//while
			
			atual->prox = novo;
		}//if
		
		L->totalIndividuos++;
	}//for
}
//=====================================================
void geraLicoes(TLista *L){
	TLicao *novo;
	int p,q;
	
	insereLicao(L, 0, 0, 0);
	insereLicao(L, 0, 1, 0);
	insereLicao(L, 1, 0, 0);
	insereLicao(L, 1, 1, 1);

}
//=====================================================
void insereLicao(TLista *L, int p, int q, int resultado){
	TLicao *novo = (TLicao *)malloc(sizeof(TLicao));
	
	novo->prox = NULL;
	novo->p = p;
	novo->q = q;
	novo->resultadoEsperado = resultado;
	
	if(L->licoes == NULL){
		L->licoes = novo;
	} else {
		TLicao *atual = L->licoes;
		
		while(atual->prox != NULL){
			atual = atual->prox;			
		}//while
		atual->prox = novo;
	}//if
}
//======================================================
void insereNeuronio(TLista *L, int neuronio){
	TNeuronio *novo = (TNeuronio *)malloc(sizeof(TNeuronio));
	novo->prox = NULL;
	novo->neuronio = neuronio;
	novo->peso = 0;
	novo->soma = 0;
	
	if(L->neuronios == NULL){
		L->neuronios = novo;
	} else {
		TNeuronio *atual = L->neuronios;
		
		while(atual->prox != NULL){
			atual = atual->prox;
		}//while
		atual->prox = novo;
	}//if
}
//======================================================
void estabelecendoSinapse(TLista *L,int neuronioDe, int neuronioAte, int camada){
	TSinapse *novo = (TSinapse *)malloc(sizeof(TSinapse));
	TSinapse *atual;
	
	novo->prox = NULL;
	novo->neuronio_origem = neuronioDe;
	novo->neuronio_destino = neuronioAte;
	novo->camada = camada;
	novo->peso = 0;
	
	if(L->pesos == NULL){
		L->pesos = novo;
	} else {
		atual = L->pesos;
		
		while(atual->prox != NULL){
			atual = atual->prox;
		}//while
		atual->prox = novo;
	}//if
}
//=============================================================
void treinamento(TLista *L){
	printf("\n\n\t\t=====| INICIADO TREINAMENTO |=====\n\n");
	fprintf(L->fp,"\n\n\tINICIO DO TREINAMENTO: ");
	//ponteiro para a struct que armazena data e hora:
	struct tm *data_hora_atual;
	//vari�vel do tipo time_t para armazenar o tempo em segundos.
	time_t segundos;
	//Obetendo o tempo em segundos.
	time(&segundos);
	//Para converter de segundos para o tempo local
	//utilizamos a fun��o localtime().
	data_hora_atual = localtime(&segundos);
	
	fprintf(L->fp,"Dia: %d", data_hora_atual->tm_mday);
	fprintf(L->fp,"   Mes: %d", data_hora_atual->tm_mon + 1);
	fprintf(L->fp,"   Ano: %d\n\n", data_hora_atual->tm_year + 1900);
	
	fprintf(L->fp,"Dia da Semana: %d.\n", data_hora_atual->tm_wday);
	
	fprintf(L->fp,"%d", data_hora_atual->tm_hour);
	fprintf(L->fp,":%d", data_hora_atual->tm_min);
	fprintf(L->fp,":%d.\n\n", data_hora_atual->tm_sec);
	
	int i;

	printf("LISTA DE ACONTECIMENTOS:\n");

	for(i= 0; i < L->Total_geracoes; i++){
		cruzamento(L, i);
		
		if((i % L->Qtd_Geracoes_para_Mutacoes) == 0){
			promoveMutacoes(L);
		}//if
		
		avaliacaoIndividuos(L);
		
		ordenamentoIndividuos(L);
		
		poda(L);
		
	}//for
	TIndividuo *atual = L->populacao;
	printf("\n\nLista ordenada por quantidade de erros:\n");
	while(atual != NULL){
		printf("individuo %d, erros: %d\n", atual->numero, atual->erros);
		atual = atual->prox;
	}
}
//=============================================================
void cruzamento(TLista *L, int flag){
	TIndividuo *filho = (TIndividuo*)malloc(sizeof(TIndividuo));
	TIndividuo *atual = L->populacao; 
	int cont, ngene;
	
	while(atual != NULL && atual->prox != NULL){
		if (flag == cont){ 
			for(ngene = 0; ngene < 4; ngene++){
				filho->genes[ngene] = atual->genes[ngene];
			}
			for(ngene = 4; ngene < 7; ngene++){
				filho->genes[ngene] = atual->genes[ngene];
			}
		}
		atual = atual->prox;
		cont++;
	}
	
	L->totalIndividuos++;
	filho->numero = flag + L->Qtd_Populacao + 1;
	atual->prox = filho; 

}
//=============================================================
void avaliacaoIndividuos(TLista *L){
    TIndividuo *atual = L->populacao;
    float n1, n2, resultadoEsperado, n3, n4, n5, soma3, peso13, peso23, soma4, peso14, peso24, soma5, peso35, peso45;
    TLicao *licao;
    
    while(atual != NULL){
        // Verifica se o indivíduo ainda não foi avaliado
        if(atual->erros == -1){            
            atual->erros = 0;
            licao = L->licoes; // Inicia a avaliação a partir da primeira lição
            
            while(licao != NULL){
                // Adequa os valores de n1 e n2 aos valores de cada lição
                n1 = licao->p;
                n2 = licao->q;
                resultadoEsperado = licao->resultadoEsperado;
                
                // Calcula n3, n4 e n5
                peso13 = atual->genes[0];
                peso23 = atual->genes[2];
                soma3 = n1 * peso13 + n2 * peso23;
                if(soma3 >= L->sinapseThreshold){
					n3 = 1;
				} else {
					n3 = 0;
				}

                peso14 = atual->genes[1];
                peso24 = atual->genes[3];
                soma4 = n1 * peso14 + n2 * peso24;
                
				if(soma4 >= L->sinapseThreshold){
					n4 = 1;
				} else {
					n4 = 0;
				}

                peso35 = atual->genes[4];
                peso45 = atual->genes[5];
                soma5 = n3 * peso35 + n4 * peso45;

                if(soma5 >= L->sinapseThreshold){
					n5 = 1;
				} else {
					n5 = 0;
				}


                // Verifica se o resultado da operação "E" não corresponde ao esperado
                if(resultadoEsperado != n5){
                    // Incrementa a quantidade de erros do indivíduo
                    atual->erros++;
                }
                
                // Avança para a próxima lição
                licao = licao->prox;
            }
        }            
        atual = atual->prox;
    }
}




//==============================================================
void ordenamentoIndividuos(TLista *L){
    TIndividuo *atual = L->populacao;
    TIndividuo *temp;
    int troca;

    do {
        troca = 0;
        atual = L->populacao;

        while (atual != NULL && atual->prox != NULL) {
            if (atual->erros > atual->prox->erros) {
                temp = atual->prox;
                atual->prox = temp->prox;
                temp->prox = atual;

                if (atual == L->populacao) {
                    L->populacao = temp;
                } 
				
				else {
                    // Encontra o nó anterior para corrigir o ponteiro de próximo
                    TIndividuo *anterior = L->populacao;
                    while (anterior->prox != atual) {
                        anterior = anterior->prox;
                    }
                    anterior->prox = temp;
                }

                troca = 1;
            }
            atual = atual->prox;
        }
    } while (troca);
}
//==============================================================
void promoveMutacoes(TLista *L){
    for(int mutacao = 0; mutacao < L->Qtd_Mutacoes_por_vez; mutacao++) {
        TIndividuo *atual = L->populacao;
        int individuo = rand() % L->totalIndividuos;
        int cont = 0;

        while (cont < individuo) {
            if (atual == NULL) {
                // Caso o índice aleatório seja maior que o número de indivíduos na lista
                printf("Erro: Índice de indivíduo aleatório excedeu o número total de indivíduos.\n");
                return;
            }
            atual = atual->prox;
            cont++;
        }

        int gene = rand() % 6;
        int sentido = rand() % 2;
        if (sentido == 0){
            atual->genes[gene] -= L->learningRate;
        } else {
            atual->genes[gene] += L->learningRate;
        }
		printf("Individuo %d sofreu mutacao no gene %d\n", atual->numero, gene);
    }
}

//==============================================================
void poda(TLista *L){
	TIndividuo *atual = L->populacao;
	while(atual->prox->prox != NULL){ //percorre a lista até o penultimo individuo
		atual = atual->prox;
	}

	atual->prox = NULL; //exclui o ultimo individuo
	free(atual->prox);
	printf("Individuo %d foi podado\n", atual->numero);

	L->totalIndividuos--;
}

//===| Programa Principal |===========================
int main(){
	inicializa(&lista);
	treinamento(&lista);
}


