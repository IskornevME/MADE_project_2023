import importlib
import metrics
# from metrics.ranking_metrics import RankingMetrics

from metrics import (
    RankingMetrics,
    Bm25,
    LaBSE,
    MsMarcoST,
    MsMarcoCE
    # USE
)

data = [{"query": ")what was the immediate impact of the success of the manhattan project?",
         "passage_text": [
             "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.",
             "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.",
             "Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.",
             "Some additional sentence",
             "Generated text",
             "fewf feofeo feioif fddss wwe"],
         "is_selected": [2, 0, 0, 0, -1, 0]},
        {"query": "what is rba",
         'passage_text': [
             "Since 2007, the RBA's outstanding reputation has been affected by the 'Securency' or NPA scandal. These RBA subsidiaries were involved in bribing overseas officials so that Australia might win lucrative note-printing contracts. The assets of the bank include the gold and foreign exchange reserves of Australia, which is estimated to have a net worth of A$101 billion. Nearly 94% of the RBA's employees work at its headquarters in Sydney, New South Wales and at the Business Resumption Site.",
             "The Reserve Bank of Australia (RBA) came into being on 14 January 1960 as Australia 's central bank and banknote issuing authority, when the Reserve Bank Act 1959 removed the central banking functions from the Commonwealth Bank. The assets of the bank include the gold and foreign exchange reserves of Australia, which is estimated to have a net worth of A$101 billion. Nearly 94% of the RBA's employees work at its headquarters in Sydney, New South Wales and at the Business Resumption Site.",
             'RBA Recognized with the 2014 Microsoft US Regional Partner of the ... by PR Newswire. Contract Awarded for supply and support the. Securitisations System used for risk management and analysis. ',
             'what is rba',
             'The inner workings of a rebuildable atomizer are surprisingly simple. The coil inside the RBA is made of some type of resistance wire, normally Kanthal or nichrome. When a current is applied to the coil (resistance wire), it heats up and the heated coil then vaporizes the eliquid. 1 The bottom feed RBA is, perhaps, the easiest of all RBA types to build, maintain, and use. 2  It is filled from below, much like bottom coil clearomizer. 3  Bottom feed RBAs can utilize cotton instead of silica for the wick. 4  The Genesis, or genny, is a top feed RBA that utilizes a short woven mesh wire.',
             'Results-Based Accountability® (also known as RBA) is a disciplined way of thinking and taking action that communities can use to improve the lives of children, youth, families, adults and the community as a whole. RBA is also used by organizations to improve the performance of their programs. RBA improves the lives of children, families, and communities and the performance of programs because RBA: 1  Gets from talk to action quickly; 2  Is a simple, common sense process that everyone can understand; 3  Helps groups to surface and challenge assumptions that can be barriers to innovation;'],
         'is_selected': [2, 3, 2, -1, 2, 3],
         }]

if __name__ == "__main__":
    # Объявление метрик
    metrics_ = [Bm25(), LaBSE(), MsMarcoCE(), MsMarcoST()]
    # Объявление класса агрегирующего обновление метрик
    rm = RankingMetrics(metrics_, [2, 3])
    for item in data:
        '''
            Обновление значений метрик, где 
            query - запрос по которому сгенерирован документ, 
            sentences - массив документов,
            labels - метки документов
        '''
        rm.update(item["query"], item["passage_text"], item["is_selected"])
        print(f"{rm.get()}\n")
        rm.show_metrics()
    """
    Пример вывода:

    Bm25_AverageLoc: 5.0   
    -----------------------------
    Bm25_AverageRelLoc: 0.83   
    -----------------------------
    Bm25_Top@1: 0.0   
    Bm25_Top@3: 0.0   
    Bm25_Top@5: 1.0   
    -----------------------------
    Bm25_FDARO@v1: 0.0   
    Bm25_FDARO@v2: 0.0   
    -----------------------------
    Bm25_UpQuartile: 0.0 
    """