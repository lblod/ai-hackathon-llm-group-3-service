import os
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
import requests


load_dotenv()
API_URL_HF = os.environ["HF_API_ENDPOINT"]
API_KEY_HF = os.environ["HF_API_KEY"]
HF_HEADERS = {"Authorization": f"Bearer {API_KEY_HF}"}


def get_hf_reply(user_prompt, system_prompt, context):
    prompt = f"<|system|>{system_prompt}<|user|>{user_prompt}\ncontext:{context}<|im_start|>assistant"

    # Payload for the API call
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL_HF, headers=HF_HEADERS, json=payload)
    return response.json()


def _relevante_artikels_prompt(pdf_tekst):
    return f"""Je bent een taalmodel dat de tekst van een uitgelezen PDF over een erfgoed ontvangt. 
        Jouw taak is om exact die delen van de tekst te selecteren die specifiek bespreken of er wel of niet wijzigingen 
        mogen worden aangebracht aan het erfgoed. Geef dit terug als een lijst object voor python.
        Met elk element in die lijst een string van een handeling waarvoor een extra toelating nodig is.
        Geef GEEN variabele naam. ENKEL DE LIJST.
        
        VOORBEELD:
        Input:
        Ministerieel besluit tot definitieve bescherming als monument van
        Architectenwoning Louis Hagen in Gent Sint-Amandsberg)
        DE VLAAMSE MINISTER VAN BUITENLANDS BELEID EN ONROEREND ERFGOED,
        Gelet op het Onroerenderfgoeddecreet van 12 JUli 2013, artikel 6.1.1;
        Gelet op het beslult van de Vlaamse Regenng van 25 JUli 2014 tot bepaling van de
        bevoegdheden van de leden van de Vlaamse Regenng, artikel 6, 1 °;
        Gelet op het m1n1steneel beslult van 21 februan 2019 tot voorlopige bescherming als
        monument van Architectenwoning Lou1s Hagen 1n Gent (Smt-Amandsberg);
        Gelet op het openbaar onderzoek dat gehouden 1s van 18 maart 2019 tot en met 16 apnl
        2019 en waarvan de behandeling 1s opgenomen 1n biJlage;
        Overwegende dat het waarderend onderzoek, waarvan de resultaten ZIJn opgenomen 1n het
        beschermmgsdoss1er, de erfgoedwaarde van Architectenwoning Lou1s Hagen aantoont;
        Overwegende dat Architectenwoning Lou1s Hagen als monument architecturale waarde bez1t
        d1e als volgt wordt gemotiveerd:
        De e1gen wonmg van architect Lou1s Hagen (1974) vormt een synthese van de
        ontwerppnnc1pes d1e de vroegste fase van ZIJn oeuvre kenmerken. D1t oeuvre bouwde Hagen
        Uit tussen 1970 en 1978, onder de vleugels van het Gentse, nationaal gerenommeerde en
        1nternat1onaal onderschelden bureau BARO. Bmnen het naoorlogse architectuurlandschap 1n
        Gent vormen de brutalistische realisaties van BARO een belangn]ke, progressieve n1che d1e
        z1ch onderscheidde van het doorsnee bouwen van dat moment. De arch1tectenwon1ng van
        Lou1s Hagen sluit hierbiJ aan en getu1gt van een hoge herkenbaarheld en representat1v1te1t,
        zowel op vlak van het behoud van de matenalltelt als van het ongmele concept.
        Op het vlak van vorm streefde Hagen er 1n ZIJn e1gen won1ng naar om te voldoen aan diverse
        randvoorwaarden, zoals biJvoorbeeld de afstemmmg op de specifieke context, met aandacht
        voor pnvacy, onentat1e en perspectieven. Deze elementen hadden een Impact op de
        ru1mtewerkmg en de Interne sch1kk1ng, d1e be1de getu1gen van een spel met contrasten en
        doorz1chten. Doorheen een ruimtelijk ontworpen wonmg met splitlevels ereeerde Hagen v1a
        een centrale trappenstructuur een architecturale wandeling. Hierdoor ontstonden diverse
        sferen en soorten ru1mtes. Het labynntacht1ge karakter van de wandeling doorheen de wonmg
        het ook een 'gliJdende schaal van pnvacy' toe. D1t kwam n1et enkel tegemoet aan noden op
        het vlak van afscherming naar de buitenwereld, maar evenzeer aan vragen tot md1v1duele
        pnvacy. Ook konden enkele ru1mtes flexibel aangepast en mgevuld worden 1n
        overeenstemming met de evoluerende gezmsnoden.
        De architectenwoning van Lou1s Hagen IS representatief voor de typologie van de
        arch1tectenwonmg en de comb1nat1e van werken en wonen. De ontvangstruimte en het
        architectuuratelier op de laagste n1veaus illustreren d1t. Hagen zette de typologie echter sterk
        naar ZIJn hand door geen ruimteliJke sche1d1ng aan te brengen tussen de representatieve en
        persoonliJke vertrekken, en te spelen met open- en geslotenheid.
        Ook de matenalen en constructiemethode droegen biJ tot de ereatle van een opt1maal woonen werkklimaat. Het ruwe, Zichtbare gebruik van betonsteen 1n het exteneur werd door Hagen
        rad1caal doorgetrokken 1n het inteneuren gecombmeerd met plafonds 1n ruw bek1ste beton.
        Pagma 1 van 8
        De matenaalkeuze is een vormeliJk statement binnen de maatschappeliJke amb1t1es van
        Hagen om z1ch door m1ddel van architectuur af te zetten tegen het matenallsme en
        kapitalisme van de toenmalige maatschappiJ. Daarnaast b1edt het onafgewerkte karakter
        ervan mogeliJkheden tot persoonliJke mbreng en toe-e1genmg van de ru1mtes door de
        bewoners.
        Alle ontwerpkeuzes leidden tot een hoge ensemblewaarde, meer bepaald een onlosmakeliJke
        verbmd1ng tussen exteneur en 1nteneur. Het gave ensemble bewaart zowel de leesbare
        matenalen en constructie, de bmnenmdellng, afwerkmg en vaste mncht1ngen 1n de keuken
        en badkamers. Het weloverwogen tumontwerp van Chnst1an Vermander (Buro voor Vn]e
        Ruimten en Groenvoorz1enmg, later Buro voor Vn]e Ru1mte) versterkte het geheel en
        mtegreerde de wonmg 1n de bu1tenaanleg. Het oorspronkeliJke tuinontwerp nam de wens tot
        pnvacy en het spel met sferen en ruimteliJk afgebakende zones Uit het mteneur over. Zo bez1t
        het volled1ge ontwerp een hoge sculpturale kwaliteit en vormt het een un1ek en erg persoonliJk
        totaal kunstwerk.
        De arch1tectenwonmg 1s echter ook representatief voor ru1mere nationale en mternat1onale
        ontwikkelingen 1n de architectuur van de late Jaren 1960 en Jaren 1970. Hagen en ZIJn
        collega's zagen- 1n navolging van het revolutieJaar 1968- 1n architectuur mogeliJkheden om
        een maatschappeliJk manifest te realiseren. D1t kon aangepast worden aan de noden van het
        gezmsleven, en nam afstand van traditionele plattegronden en won1ngtypes.
        Zo leunt ze zowel m de vormeliJke als conceptuele keuzes nauw aan biJ het Nederlandse
        structuralisme van Aldo Van Eyck en Herman Hertzberger, en de ontwerppnnc1pes van Lou1s
        Kahn. Illustratief hiervoor ZIJn de aandacht voor de menseliJke schaal en de persoonliJke
        mbreng van gebruikers en bewoners, gestimuleerd door de specifieke Indeling en Zichtbare
        matenalen. De ontwerpen vormen een metafoor van een kle1ne stad, met aandacht voor
        bmnenstraten, ru1mtes voor ontmoetmg, voor persoonliJke ontwikkeling en voor beschutting.
        De complexe, fragmentansche ruimteliJkheld van de ontwerpen VIsualiseert de contrasten d1e
        de architect w1l verzoenen. De opvallende ruimteliJk structurerende constructie breekt met
        trad1t1onele conventles en legt de focus op het wonen zelf. Deze structuur laat bovend1en
        · flexib1l1te1t toe op het vlak van gebruik en mvullmg.
        Hagens arch1tectenwonmg kwam ook tot stand op het moment dat het Gentse brutalisme een
        hoogtepunt beleefde. Het ontwerp vormt een VISitekaartJe en modelproJect van de filosofie
        van BAROen van de ontwerppnnc1pes d1e Hagen zelf neerschreef. De open ru1mtewerkmg en
        de zichtbare matenalen vertonen parallellen met het naoorlogse brutalisme van Hagens
        collega's en vnenden biJ BARO, Schaffrath en Raman, en tiJdgenoten als Jullaan Lampens en
        Marc Dessauvage, en verwiJZen eveneens naar pnnc1pes u1t het mternat1onale modernisme,
        BESLUIT:
        Artikel 1. Met toepassing van art1kel 6.1.1 tot en met art1kel 6.1.11 van het
        Onroerenderfgoeddecreet van 12 JUli 2013 en art1kel 6.2.1 van het Onroerenderfgoedbesluit
        van 16 me1 2014 worden de volgende onroerende goederen defm1t1ef beschermd als
        monument:
        Architectenwoning Lou1s Hagen met tum, N1Jverhe1dskaa1 43 1n Gent (Smt-Amandsberg),
        bekend ten kadaster: Gent, 19de afdeling, sect1e C, perceelnummer 1225Z.
        De defin1t1ef beschermde onroerende goederen ZIJn aangeduld op het plan dat als biJlage biJ
        d1t beslult wordt gevoegd.
        De fotoreg1strat1e van de fys1eke toestand van de defm1t1ef beschermde goederen wordt als
        biJlage biJ d1t beslult gevoegd.
        Art. 2. Het monument heeft architecturale waarde.
        Pag1na 2 van 8
        De erfgoedelementen en de erfgoedkenmerken van het monument ZIJn:
        Inplantmg en tuinaanleg
        De architectenwoning ligt 1n een verkaveling ten zuidoosten van de kern van SmtAmandsberg, aan de NiJVerheidskaal. Deze straat IS enkel aan de noordziJde bebouwd. Ten
        zu1den wordt ze afgeliJnd door de Schelde.
        De 1nplantmg van Hagens wonmg vertrok vanuit de aandacht voor pnvacy van de bewoners.
        De groenaanleg schermt de wonmg af van de straat en Integreert de architectuur m de
        omnngende natuur. Het ongmele tumontwerp door landschapsarch1tect Chnst1an Vermander
        IS op z1ch bewaard, maar de oorspronkeliJke structuur IS mlnder herkenbaar. Het ontwerp
        speelde aanvankeliJk sterk m op het grondplan van de won1ng. De verharde opnt tot de
        garage 1s uitgewerkt met een baJonetvormige asverschu1vmg, waarop tegeliJk een smal en
        vnJliggend verhard mkompad naar de voordeur 1n de westgevel leidt. Deze padenstructuur m
        betonklmkers 1s bewaard.
        De voortumru1mte bewaart referentles aan het oorspronkeliJke ontwerp met compacte, m1n
        of meer rechthoekige tumkamers, afgescheiden door gesloten heesterwanden en enkele
        verspreide opgaande bomen. De haagbeukenhagen d1e de tuinkamers begrensden, resteren
        momenteel enkel nog aan de ZIJde van de opnt en de straat. Ter hoogte. van de wonmg slUit
        een verhard 'b1nnenple1nt]e' aan met 'verzonken z1tru1mte' en een hoekJe aansluitend biJ een
        ronde betonnen waterbak. De licht verdiepte Zithoek 1s heden bewaard m de vorm van een
        vlerkante VIJVer. De tumaanleg staat m dialoog met een verhoogd terras boven de garage.
        D1t verhard terras verdeelde z1ch m een 'eetkamer' m open lucht en een 'zonneterras',
        Ingekleed met plantenbak. De afwatenng voert met een druipketting af m de geliJkvloerse
        waterbak. De tumkamers werden dooraderd door een speels tracé van stapstenen Uit
        hergebruikt gramet.
        De ondiepe achtertuinruimte met beperkte terremmodellenng 1s losser vormgegeven. Ook
        deze tumrUimte, deels opgevat als bloemenweide, deels als gesloten heesterwand, met
        enkele opgaande bomen 1s dooraderd door een speels tracé van stapstenen dat verbmdmg
        maakt met de 'mkom' en de 'verzonken z1tru1mte'.
        In de tu1n resteren momenteel nog een beperkt aantal beplantmgen van de oorspronkeliJke
        aanleg, waaronder enkele klimplanten als Wilde wmgerd en kamperfoelie. De architectuur
        mtegreert z1ch sterk m de bu1tenru1mte door de aanwezigheld van deze planten. Het was
        1mmers de wens van de architect om door middel van de begroe11ng van de won1ng de breuk
        tussen de natuur en de architecturale 1ngreep te verzachten. In de struiklaag ZIJn biJVoorbeeld
        de krentenboompJes bewaard.
        Exteneur
        De wonmg 1s opgevat als een kubusvormig volume onder een plat dak. De buitenarchitectuur
        ont- en verhult op subtiele WIJZe de mwend1ge structuur op bas1s van splitlevels, d1e aan de
        straatziJde (zuid) twee bouwlagen met verspnngende n1veaus voorziet en achteraan (noord)
        dne n1veaus. De westgevel wordt verlevendigd door m- en uitsprongen van het volume. De
        zuidZIJde wordt op straatniveau u1tgebre1d met een volume van één bouwlaag, dat dienst doet
        als garage (garagepoort ten zuiden) of als overdekte bu1tenru1mte (schuifraam m de
        oostZIJde). Daarboven IS een Uitbouw met terras voorzien, deels gesitueerd onder een luifel,
        gevormd door een Ultkragmg van het dak.
        De geslotenheid van de architectuur wordt versterkt door het matenaalgebru1k. Het parement
        bestaat u1t zichtbare betonstenen. Tegen de oostziJde van de garage ZIJn een betonnen
        waterspuwer en c11indervorm1ge waterput voorzien. Een metalen kettmg leidt het regenwater
        af tot 1n de waterput. U1tspnngende muurtJes m betonsteen biJ de garage en het
        bovenliggende terras verhogen eveneens de sculpturaliteit van het geheel.
        De aanwezige muuropeningen stemmen overeen met de planindeling, de onentat1e en de
        verzoenmg tussen afschermmg en contact. De noord- en westgevel ZIJn grotendeels gesloten
        uitgewerkt, maar ook de vensteropenmgen van de andere twee gevels ZIJn weloverwogen
        aangebracht m relatle tot de bu1tenru1mte. De rechthoekige vensters spnngen licht terug ten
        Pagma 3 van 8
        opzichte van het parement m betonsteen en bewaren quas1 volledig hun oorspronkeliJk
        houten schn]nwerk.
        De zuidgevel 1s het sterkst opengewerkt. De bovenverd1epmg wordt ter hoogte van het terras
        verlicht door een groot schuifraam, dat eveneens de toegang tot het terras vormt. Ernaast,
        ten oosten ervan, 1s de gevel volled1g opengewerkt met een venster dat de ontvangstruimte
        en bovenliggende Zithoek verlicht en doorloopt over de hoek met de oostgeveL De oostgevel
        wordt centraal doorbroken door een verticaal, gevelhoog venster, dat schum afloopt
        bovenaan tot het dak. De noordoosteliJke ZIJde van deze gevel IS voorz1en van een
        hooggeplaatst venster ter hoogte van de slaapkamer en van twee klemere, dieperliggende
        hoekvensters. De noordgevel wordt behalve door de hoekvensters verlicht door twee
        asymmetnsch 1n de gevel geplaatste vensters. Het gesloten karakter van de noordgevel wordt
        verdergezet 1n de Uitbouw van de westgeveL Een hoekvenster verlicht de keuken en stroken
        met glasdallen ZIJn voorz1en ter hoogte van het to1let en de douchekamer.
        De toegang IS verdiept gesitueerd m de westgeveL De oorspronkeliJke houten deur met een
        honzontale beplanking 1s bewaard en wordt lmks geflankeerd door een ZIJlicht. In de
        zu1dgevel, rechts naast de u1tbouw met garage, 1s eveneens een toegangsdeur voorz1en, m
        d1t geval een beglaasde deur met bovenlicht.
        De combmat1e van de opengewerkte bovenverd1epmg met aansluitend terras aan de zuidkant
        van de wonmg speelt volledig m op het panorama. Toch wordt InkiJk vermeden. Het terras 1s
        ommuurd en ten zu1den voorz1en van een eveneens ommuurde 'groenbuffer', geflankeerd
        door een 'wmdbeschutte zone'. Aan de westZIJde van het terras bev1ndt z1ch een Ingemaakte
        kast, mgepast m een verticale u1tbouw m betonsteen m de westgevel.
        Inteneur
        Planmdeling en algemene kenmerken:
        De planmdeling en mteneurafwerkmg kunnen n1et losgekoppeld worden van het exteneur.
        Hagen trok 1n het mteneur het brutal1st1sche karakter door en liet de betonstenen wanden en
        constructleve elementen m ruw Zichtbeton of 'béton brut' overal zichtbaar. De specifieke
        constructiemethode leidde tot een complex en weloverwogen ontwerp waarbiJ alles op
        voorhand gepland d1ende te worden en le1dmgen gemtegreerd en verborgen werden. De
        verlichtingsarmaturen werden mgepast 1n daarvoor voorziene, c1rkelvorm1ge u1tspanngen in
        de betonnen plafonds.
        Achter de gesloten façade gaat een dynamische ru1mtewerkmg schuil volgens een open plan.
        De bewoners en gebrulkers kunnen het gebouw stelselmatig ontdekken, vanu1t de donkere
        benedenverd1epmgen tot de lichte leefruimtes op de bovenverd1epmgen. De lichtmval wordt
        bepaald door de opengewerkte zuidgevel en een gevelhoog venster en v1de m de oostgeveL
        Zenitale verlichtmg IS voorz1en v1a een lichtkoepelm de badkamer. De ru1mtes ZIJn gesitueerd
        op meerdere splitlevels, wat zorgt voor een ruimteliJke complex1te1t rond de centrale traphal.
        De bordestrap ontvouwt z1ch rondom een verticaal structurerend, sculpturaal element, waann
        het gebruik van betonsteen wordt verdergezet. De trap en de piJlers m betonsteen d1e deze
        flankeren, spelen met openheld en geslotenheid. De opengewerkte trap voorz1et eveneens
        VriJe doorgangen tot de n1veaus d1e erop aansluiten. Ook tussen de mveaus onderling wordt
        gespeeld met v1suele relaties en de grens tussen afschermmg/pnvacy en openheid/contact.
        Weloverwogen contact met de buitenruimte maakt het ensemble compleet.
        De aandacht van Hagen voor het wonen en het gezmsleven leidde tot het verzachten van de
        brute matenalen door de vloeren van verschillende ru1mtes en de betonnen trap met tapiJt te
        bekleden. Deze aandacht zorgde er ook voor dat de sfeer en bestemmmg van de ru1mtes
        bepalend was voor hun s1tuenng, aankleding en hoogte. De mt1emere offunct1onelere ru1mtes
        werden aan de achterziJde (noord) gesitueerd, wat daar leidde tot dne lagere n1veaus. De
        representatieve leefru1mtes, gesitueerd m twee hogere n1veaus aan de straatZIJde (zu1d),
        voorzagen telkens nog een n1veauversch1l, wat de dynamiek van de ru1mtewerk1ng
        verhoogde.
        Pagma 4 van 8
        Benedenverd1epmgen:
        De toegang 1n de westgevel le1dt v1a een open tochtportaal met mgewerkte mat, tot de hal
        (n1veau 0). In de noordwand van het portaal 1s een u1tspanng aanwezig, waartussen een
        bewaarde afschermmg van de verwarmmg en een legplank ZIJn voorz1en.
        De ru1me, open hal IS voorzien van een vloer m spliJttegels 1n rood aardewerk en b1edt ook
        aan de ZUidZiJde een toegang tot de tum. In de zuidwesteliJke hoek IS een tollet gesitueerd,
        evenals een toegang tot de garage. Deze paneeldeuren ZIJn ongineel en op de houten liJsten
        na voorzien van een blauwe afwerking. De deuren ten noordwesten van de hal d1e toegang
        geven tot een douchekamer en stookplaats, ZIJn geliJkaardig Uitgewerkt. In het tollet en de
        douchekamer wordt de betegelde vloer van de hal doorgetrokken. De douchekamer bewaart
        eveneens het lavabomeubel m zwarte form1ca met twee lavabo's, voorz1en van een zwarte
        afwerking en geplaatst tegen de westwand. Ook bewaart de noordeliJke doucheruimte met
        z1tbad de ong1nele betegeling met blauwe moza1eksteent]es.
        De hal 1s met een betonstenen muur afgesloten van de twee treden lager gelegen
        ontvangstruimte (n1veau -1). De lage muur zorgt wel voor een open verbmd1ng tussen de
        ru1mtes. De betegelde vloer van de hal wordt doorgetrokken tot de trap en de vloer van de
        ontvangstruimte. Een lage betonstenen muur schermt de noordziJde van deze ru1mte af, maar
        opent deze terzelfder tiJd naar de achterliggende v1de en het laagste n1veau.
        Enkele treden, eveneens afgewerkt met de spli]ttegels, geven toegang tot het laagste n1veau
        aan de noordZIJde van de wonmg (n1veau -2). Dit laagste n1veau IS voorzien van een vloer
        met functionele, keram1sche tegels. Het had geen vastgelegde bestemmmg, kon worden
        aangepast aan de noden van het gezm en deed vanaf 1978 d1enst als tekenatelier. VlakbiJ de
        treden bev1ndt z1ch een bewaarde deur met een blauwe afwerkmg, d1e le1dt tot de kruipkei der.
        Ten oosten IS een smalle ru1mte voorz1en 1n de vide, d1e m een open verbmdmg staat met de
        hoger gelegen ontvangstruimte en verlicht wordt door het gevelhoge venster m de oostgeveL
        Bovenverd1epmgen:
        Vanu1t de hal kunnen gebrulkers en bezoekers ook het parcours naar boven volgen. De
        bordestrap 1s vanaf daar bekleed met tapiJt en le1dt 1n eerste mstant1e naar een 1nt1eme ru1mte
        aan de noordZIJde (mveau 1). Het lage, relatief donkere karakter van de ru1mte wordt
        versterkt door de beperkte muuropeningen en de vloer, d1e met tapiJt 1s afgewerkt. Ten
        noordwesten 1s ru1mte voor 'bib en studie' voorz1en en ten noordwesten een '1nt1eme Zithoek'.
        De l1chtmval wordt grotendeels bepaald door het spel met open- en geslotenheid van de v1de
        ten zu1den en de doorzichten tot de trap. De z1thoek 1s enkel voorz1en van een subtiele
        afslu1t1ng m de vorm van een lage betonstenen muur. Ten zu1den 1s de Zithoek opengewerkt
        tot de v1de en de bovenliggende leefruimte. Voor de v1de 1s er een haardensemble voorzien,
        opgebouwd Uit betonnen elementen, d1e de geometnsche, sculpturale opbouw van de
        architectuur verderzetten. Het tablet ten oosten IS afgedekt met tegels m aardewerk, en wordt
        ten westen ervan geflankeerd door een verdiepte zone met de aslade, afgeliJnd aan de
        voorZIJde met baksteen. De haard zelf combmeert een betonnen balkvormig volume met drie
        c11indervorm1ge verluchtlngspiJpen aan de voorziJde en een zichtbare, Cilindervormige schouw
        tot het dak.
        De trap le1dt vervolgens naar de leefruimtes aan de zuidZIJde. In eerste mstant1e 1s d1t de
        open z1thoek (n1veau 2). Deze 1s n1et alleen ten zu1den en zuidoosten geopend naar de
        omgevmg. Ze staat ook ten noorden m verb1ndmg met de v1de, de haard, de lager gelegen
        z1thoek en het bovenliggende n1veau met het nachtgedeelte. Ook het gevelhoge venster 1n
        de oostgevel ter hoogte van de v1de ereeert lichtmval. Het hoge plafond, het sculpturale spel
        met betonstenen wanden en piJlers, versterkt biJkomend het ruimtegevoeL
        Met tapiJt beklede treden le1den vervolgens tot de eethoek en open keuken aan de
        zuidwestZIJde (n1veau 3), d1e ten zulden m verb1ndmg staat met het terras. Deze ru1mtes ZIJn
        net als de benedenverdieping voorzien van een vloer 1n spli]ttegels. De volledige
        keukenmnchtmg 1n form1ca 1s ongmeel en voorz1en van een bar aan de ZIJde van de eethoek.
        Het keukenmeubilair werd net als de badkamermeubels gerealiseerd door Wilfra Keukens
        (Meulebeke). De kasten en werkbladen spelen met een contrast tussen de zwarte hoofdkleur
        Pagma 5 van 8
        en w1tte accenten. De architect koos hiervoor, aangezien w1t een te dommante kleur zou ZIJn
        1n de woonsfeer. Ook de wandafwerking met donkergn]ze moza1eksteent]es 1s bewaard.
        Het nachtgedeelte met badkamer en slaapkamer, dat z1ch op het hoogste n1veau aan de
        noordziJde bevmdt (n1veau 4). De bad- en slaapkamer staan m een open verbmd1ng met
        elkaar en ZIJn voorz1en van een vloer m tapiJt. De badkamer (noordwest) bewaart de blauwe
        moza1ekbetegelmg van het bad en de wanden, evenals het donkere lavabomeubel en de
        kasten d1e een sche1dmg vormen tot de 'alkoof'. Een Vlerkante openmg m het plafond, d1e het
        beton zichtbaar toont en afgesloten 1s met een koepel, verlicht de badkamer. De slaapn1s
        (noordoost) kn]gt licht v1a een venster m de oostwand en vanu1t het mwend1ge doorzicht ten
        zu1den.
        Art. 3. Voor het beschermde monument gelden de volgende beheersdoelstellingen:
        1 o de algemene doelstelling van de beschermmg IS het behoud van de erfgoedkenmerken
        en -elementen d1e de bas1s vormen voor de erfgoedwaarde;
        2° Arch1tectenwonmg Lou1s Hagen vormt een totaalconcept met een belangnJke relatle
        tussen het architectuurontwerp en de aanleg van de bu1tenru1mte. Het pand getu1gt van
        een onlosmakeliJke samenhang tussen exteneur, mplantmg en mteneur. Elke
        beheersdaad vraagt om een ge1ntegreerde en duurzame aanpak waarbiJ de Impact op
        de volledige s1te met al z'n componenten wordt afgewogen. Ingrepen d1enen de
        beschermde erfgoedkenmerken en -elementen te respecteren en te ondersteunen, en
        mogen de draagkracht van de beschermde s1te n1et overschn]den. D1t veronderstelt
        vakkundig onderhoud en ind1en nod1g conserverende ingrepen;
        3° met betrekkmg tot het exteneur van Arch1tectenwonmg Lou1s Hagen beoogt de
        beschermmg het behoud van het oorspronkeliJke bouwvolume qua schaal,
        gevelopbouw, afwerkmg, matenaalgebruik en schn]nwerk, d1e bepalend ZIJn voor de
        architecturale e1genhe1d en herkenbaarheld van het ontwerp. Ind1en behoud en herstel
        n1et meer mogeliJk 1s, dient het schn]nwerk vervangen te worden naar engmeel model
        met respect voor het matenaal en de geledingen van het ong1nele houten schn]nwerk;
        4° met betrekking tot het 1nteneur beoogt de beschermmg het behoud van de planmdelmg,
        de open ru1mtewerkmg, de Vides, de kenmerkende matenalen en afwerkmg van de
        muren, vloeren en de plafonds, bmnenschn]nwerk, en de bewaarde vaste 1nnchtmg van
        de badkamers, keuken en Zithoek met haard. U1terst bepalend 1s de geled1ng van de
        ru1mte met splitlevels, afgescheiden door lage muurtJes en 1n een open verb1ndmg met
        de centrale trappenpartiJ en aanwezige v1des; ·
        5° het beeldbepalende, Zichtbare karakter van de matenalen (beton(steen)) d1ent steeds
        behouden te bliJven en versterkt de samenhang tussen mteneur en exterieur. Enkel m
        de twee kamers aan de noordZIJde van het laagste n1veau, waar d1t matenaal alm 1978
        werd geschilderd, 1s er een beperktere v1suele Impact op het geheel en kan h1er vn]er
        mee worden omgegaan; .
        6° de recentere aanpassmg van het dak, het n1euwe dakvolume en de draaltrap tegen de
        westgevel maken geen deel u1t van het oorspronkeliJke ontwerp. Toekomstig behoud 1s
        dus n1et vere1st en een terugkeer naar de oorspronkeliJke toestand 1s steeds mogeliJk.
        Aangezien de toevoegmgen n1et 1ngn]pen op de matene vormen ze wel reversibele
        mgrepen;
        7° de beschermmg van het volled1ge perceel met mbegnp van de tUin veronderstelt een
        respectvolle omgang met het bomenbestand en de aanwez1ge aanplantmgen. Het
        vere1st ook het behoud van de weloverwegen aanleg, voor zover bewaard. Het IS
        wenseliJk dat de tumaanleg 1n de toekomst zoveel mogeliJk wordt hersteld 1n de geest
        van het oorspronkelijke ontwerpplan, dat getu1gt van een grote samenhang en dialoog
        met de architectuur en b1nnenru1mtes.
        Art. 4. De zakeliJkrechthouder en de gebruiker van het beschermde monument ZIJn verplicht
        de mstandhoudmg en het onderhoud ervan te verzekeren door:
        1° het goed als een goede hulsvader te beheren en de nod1ge voorzorgsmaatregelen te
        nemen tegen schade ten gevolge van brand, blikseminslag, diefstal, vandalisme, w1nd
        of water;
        2° de toestand van het goed regelmatig te controleren;
        Pagma 6 van 8
        3° regulier onderhoud u1t te oefenen;
        4° onmiddelliJk passende consolidatie- en beve11igmgsmaatregelen te nemen 1n geval van
        nood.
        Art. 5. Voor de volgende handelingen aan het beschermde monument moet een toelat1ng
        worden aangevraagd:
        1 o het plaatsen, slopen, verbouwen of heropbouwen van een constructie;
        2° het verwijderen, vervangen, wijZigen of verstevigen van constructleve elementen;
        3° het verwijderen, vervangen of wijZigen van h1stonsche matenalen en het toepassen van
        behandelingen met als doel de h1stonsche matenalen te rem1gen, te herstellen, te
        verduurzamen of te beschermen tegen verweer en aantasting;
        4° het Uitvoeren van de volgende werken aan het dak en de buitenmuren van constructies:
        a) het verwijderen, vervangen of wijZigen van dakbedekking en gootconstructies;
        b) het verwiJderen van voegen en het hervoegen;
        c) het aanbrengen, verWIJderen, vervangen of WIJZigen van de kleur, textuur of
        samenstelling van de afwerkmgslagen;
        d) het aanbrengen, verWIJderen, vervangen of WIJZigen van bultenschn]nwerken,
        deuren, ramen, luiken, poorten, mclus1ef de al dan n1et f1gurat1eve beglazmg,
        claustra, beslag, hang- en sluitwerk;
        e) het aanbrengen, verWIJderen, vervangen of WIJZigen van aard- en nagelvaste
        elementen, smeediJZer en beeldhouwwerk, 1nclus1ef n1euwe toevoegingen;
        f) het aanbrengen, vervangen of WIJZigen van opschnften, pubiiC1te1tsmnchtmgen of
        uithangborden, met u1tzondenng van verk1ezingspubllc1te1t en met u1tzondenng van
        publlclteltsmnchtmgen, waarbiJ wordt bekendgemaakt dat het goed te koop of te
        huur 1s, op voorwaarde dat de totale max1male oppervlakte n1et meer bedraagt dan
        4 m2 ;
        5° het u1tvoeren van de volgende omgevmgswerken:
        a) het plaatsen of WIJZigen van bovengrondse nutsvoorz1en1ngen en le1d1ngen;
        b) het plaatsen of WIJZigen van afslu1t1ngen, met u1tzondenng van gladde schnkdraad
        en pnkkeldraad ten behoeve van veekenng;
        c) het aanleggen, structureel en fundamenteel WIJZigen of verWIJderen van wegen en
        paden;
        d) het vellen of beschadigen van bomen en struiken d1e opgenomen ZIJn 1n het
        beschermmgsbeslu1t of 1n een goedgekeurd beheersplan, en elke handeling d1e een
        WIJZiging van de groeiplaats en groe1vorm van de bomen en de stru1ken d1e
        opgenomen ZIJn 1n het beschermmgsbeslUit of 1n een goedgekeurd beheersplan tot
        gevolg kan hebben;
        e) het aanleggen of WIJZigen van verhardmg met een m1mmale gezamenliJke
        grondoppervlakte van 30 m2 of het u1tbre1den van bestaande verhardingen met
        mmimaal 30 m2 , met u1tzondenng van verhardmgen geplaatst bmnen een straal
        van 30 m rond een vergund of een vergund geacht gebouw;
        f) het aanleggen van sport- en spelinfrastructuur of parkeerplaatsen;
        g) het structureel en fundamenteel w1jz1gen van de aanleg van de tum;
        6° het Uitvoeren van de volgende handelingen aan of m het 1nteneur:
        a) het Uitvoeren van destructief matenaaltechmsch onderzoek;
        b) het u1tvoeren van structurele werken en het toevoegen van meuwe structuren;
        c) het verWIJderen, vervangen of WIJZigen van h1stonsche matenalen en het toepassen
        van behandelingen met als doel de h1stonsche matenalen te rem1gen, te herstellen,
        te verduurzamen of te beschermen tegen verweer en aantasting;
        d) het verwiJderen, vervangen of WIJZigen van plafonds, gewelven, vloeren, trappen,
        bmnenschn]nwerken, 1nclus1ef de al dan n1et f1gurat1eve beglazing, lambnsenng,
        beslag, hang- en sluitwerk, en van de waardevoile mteneurdecorat1e;
        e) het bepleisteren van n1et-beple1sterde elementen of het bepleisteren met een
        andere samenstellmg of textuur, alsook het ontple1steren van bepleisterde
        elementen;
        f) het beschilderen van ongesch1lderde elementen of het schilderen 1n andere kleuren
        of kleurschakenngen of met een andere verfsoort dan de aanwez1ge;
        Pagma 7 van 8
        g) het plaatsen of vernieuwen van technische voorzieningen zoals verwarming,
        klimaatregeling, elektrische installatie, geluidsinstallatie, sanitair, liften en
        beveiligingsinstallaties, met uitzondering van die installaties waarvoor geen
        destructieve ingrepen moeten gebeuren en/of die geen storende visuele impact
        hebben op de erfgoedelementen en -kenmerken.
        Er is geen toelating vereist voor het onmiddellijk nemen van passende consolidatie- en
        beveiligingsmaatregelen in geval van nood, noch voor de uitvoering van regulier onderhoud.
        Brussel, 2 9 MEI 2019
        De Vlaamse minister van Buitenlands Beleid en Onroerend Erfgoed,
        Geert BOURGEOIS
        Pagina 8 van 8 
        
        Output:
        ["het plaatsen, slopen,  verbouwen of heropbouwen van  een  constructie",
        "het verwijderen,  vervangen, wijzigen  of verstevigen  van  constructleve elementen",
        "het verwijderen, vervangen of wijzigen  van h1stonsche matenalen en  het toepassen van behandelingen  met  als  doel  de  h1stonsche  matenalen  te  rem1gen,  te  herstellen,  te verduurzamen of te beschermen  tegen  verweer en  aantasting",
        "het uitvoeren van de volgende werken aan  het dak en  de buitenmuren van constructies: ",
        "het verwijderen,  vervangen of wijzigen  van  dakbedekking en  gootconstructies",
        "het verwiJderen  van  voegen  en  het hervoegen",
        "het aanbrengen,  verwijderen, vervangen  of  wijzigen van de kleur,  textuur  of samenstelling van  de  afwerkmgslagen",
        "het aanbrengen,  verwijderen, vervangen  of  wijzigen van bultenschn]nwerken, deuren,  ramen,  luiken,  poorten,  mclus1ef  de al dan n1et f1gurat1eve beglazmg, claustra,  beslag,  hang- en  sluitwerk",
        "het aanbrengen,  verwijderen, vervangen  of  wijzigen van aard- en nagelvaste elementen, smeediJZer en  beeldhouwwerk, 1nclus1ef n1euwe  toevoegingen",
        "het aanbrengen,  vervangen  of wijzigen  van  opschnften,  pubiiC1te1tsmnchtmgen  of uithangborden, met u1tzondenng van verk1ezingspubllc1te1t en  met u1tzondenng van publlclteltsmnchtmgen,  waarbiJ  wordt bekendgemaakt dat het goed  te  koop  of te huur 1s,  op  voorwaarde dat de totale max1male oppervlakte n1et  meer bedraagt dan 4  m 2 ",
        "het uitvoeren  van de volgende omgevingswerken: ",
        "het plaatsen  of wijzigen  van  bovengrondse nutsvoorz1en1ngen  en  leidingen",
        "het plaatsen  of wijzigen  van  afslu1t1ngen,  met u1tzondenng  van  gladde schnkdraad en  pnkkeldraad ten  behoeve van  veekenng",
        "het aanleggen,  structureel  en  fundamenteel  wijzigen  of verwijderen  van wegen  en paden",
        "het vellen of  beschadigen  van bomen  en struiken  d1e opgenomen  ZIJn 1n het beschermmgsbeslu1t of 1n  een  goedgekeurd beheersplan, en  elke handeling d1e  een wijZiging van de groeiplaats  en groe1vorm van de bomen en de stru1ken d1e opgenomen ZIJn  1n  het beschermmgsbeslUit of 1n  een  goedgekeurd  beheersplan tot gevolg  kan  hebben",
        "het aanleggen of  wijZigen van verhardmg met een m1mmale gezamenliJke grondoppervlakte  van  30  m 2  of het  u1tbre1den  van  bestaande  verhardingen  met mmimaal  30  m 2 , met u1tzondenng  van  verhardmgen  geplaatst  bmnen  een  straal van  30  m  rond  een  vergund  of een  vergund geacht gebouw",
        "het aanleggen  van  sport- en  spelinfrastructuur of parkeerplaatsen",
        "het structureel  en  fundamenteel  w1jz1gen  van  de aanleg van  de tum",
        "het Uitvoeren van de volgende handelingen  aan  of m het 1nteneur: ",
        "het Uitvoeren van destructief matenaaltechmsch onderzoek",
        "het uitvoeren  van  structurele werken  en  het toevoegen  van  nieuwe structuren",
        "het verwijderen, vervangen of wijZigen  van  historische materialen en  het toepassen van  behandelingen met als doel de h1stonsche matenalen te rem1gen,  te herstellen, te verduurzamen of te  beschermen tegen  verweer en  aantasting",
        "het verwiJderen,  vervangen  of wijZigen  van  plafonds,  gewelven, vloeren, trappen, beschermingswerken,  1nclus1ef  de  al dan  n1et  f1gurat1eve  beglazing,  lambnsenng, beslag,  hang- en  sluitwerk, en  van de waardevoile mteneurdecorat1e",
        "het bepleisteren  van n1et-beple1sterde elementen  of  het  bepleisteren met  een andere samenstelling of  textuur, alsook het ontple1steren van bepleisterde elementen",
        "het beschilderen van ongesch1lderde  elementen  of het schilderen 1n  andere kleuren of kleurschakenngen of met een  andere verfsoort dan  de  aanwez1gePagma  7 van  8 ",
        "het plaatsen of  vernieuwen van technische voorzieningen zoals verwarming, klimaatregeling, elektrische installatie, geluidsinstallatie, sanitair, liften en beveiligingsinstallaties, met uitzondering van die installaties waarvoor geen destructieve  ingrepen  moeten  gebeuren  en/of die  geen  storende  visuele  impact hebben  op  de  erfgoedelementen  en  - kenmerken."]
        EINDE VOORBEELD
        """


def get_toelatingsplichtige_handelingen(besluit: list[Document]) -> List[str]:
    context = "\n".join([doc.page_content for doc in besluit])
    user_prompt = _relevante_artikels_prompt()
    reply = get_hf_reply(user_prompt=user_prompt,
                         system_prompt="",
                         context=context)
    return reply[0]["generated_text"]
