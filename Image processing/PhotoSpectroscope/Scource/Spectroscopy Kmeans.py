# Modules Needed

import os
import cv2

import math
import numpy as np

from matplotlib import colors
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Main Code Starts Here


def capture():
    cap = cv2.VideoCapture(0)
    roi_selected = False
    i_dist = []
    while True:
        ret, frame = cap.read()
        k = cv2.waitKey(1)

        if k & 0xFF == ord("s") and roi_selected:
            shape = cropped.shape
            for i in range(shape[1]):
                r_val = np.mean(cropped[:, i][:, 0])
                g_val = np.mean(cropped[:, i][:, 1])
                b_val = np.mean(cropped[:, i][:, 2])
                i_val = (r_val + g_val + b_val) / 3
                i_dist.append(i_val)

        elif k & 0xFF == ord("r"):
            r = cv2.selectROI(frame)
            roi_selected = True

        elif k & 0xFF == ord("q"):
            break

        else:
            if roi_selected:
                cropped = frame[
                    int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])
                ]
                cv2.imshow("ROI", cropped)
            else:
                cv2.imshow("FRAME", frame)

    cap.release()
    cv2.destroyAllWindows()
    return i_dist


def normalise(spectrumIn):

    spectrumOut = []

    maxPoint = max(spectrumIn)

    for value in spectrumIn:
        spectrumOut.append(value / maxPoint)

    return spectrumOut


def transmittance(reference, sample):

    # Calculate Transmittance

    transmittance = []
    absorbance = []
    reflectance = []

    for i in range(len(sample)):
        if sample[i] == 0:  # This 'If' Is To Avoid Division By Zero Error
            transmittance.append(0)
            absorbance.append(0)
            reflectance.append(0)

            # Conceptually Wrong, If Sample > Reference, Artificious Data Distortion Has Happened

        else:
            transmittance.append(sample[i] / reference[i])
            absorbance.append(-math.log(transmittance[i], 10) / 5)
            reflectance.append(
                1 - (sample[i] / reference[i]) + (-math.log(transmittance[i], 10) / 5)
            )

    return (
        transmittance  # Returns Reflectance - Change From Here For The Main Driver Code
    )


def absorbance(reference, sample):

    # Calculate Absorbance

    absorbance = []

    for i in range(len(sample)):
        if sample[i] == 0:  # This 'If' Is To Avoid Division By Zero Error
            absorbance.append(0)
        else:
            absorbance.append(-math.log(sample[i] / reference[i], 10) / 5)

    return absorbance


def reflectance(reference, sample):

    # Calculate Reflectance

    reflectance = []

    for i in range(len(sample)):
        if sample[i] == 0:  # This 'If' Is To Avoid Division By Zero Error
            reflectance.append(0)
        else:
            reflectance.append(
                1
                - (sample[i] / reference[i])
                + (-math.log(sample[i] / reference[i], 10) / 5)
            )

    return reflectance


pixel = [115, 146, 193, 250, 312, 329, 404]
wavelength = [405.4, 436.6, 487.7, 546.5, 611.6, 631.1, 708]
reference = [
    0.022867116398281522,
    0.022867116398281522,
    0.01939870834350586,
    0.019590424961513943,
    0.019015266100565593,
    0.015285412470499674,
    0.012993472483423021,
    0.010736390948295593,
    0.009751642710632748,
    0.010649246672789256,
    0.011555566953288186,
    0.014143802656067744,
    0.017620927890141806,
    0.018100232283274332,
    0.017176488902833728,
    0.013673213455412123,
    0.011398699482282003,
    0.010849680569436816,
    0.010379093289375306,
    0.010431379278500874,
    0.010318091478612687,
    0.010675388740168677,
    0.0105359575814671,
    0.011442275775803458,
    0.010631819069385528,
    0.010466241472297244,
    0.010535958376195698,
    0.011111123065153758,
    0.011503279805183411,
    0.010413953496350182,
    0.009847502741548749,
    0.00895861311091317,
    0.009446631438202327,
    0.010239663687017229,
    0.010727679398324754,
    0.010065367817878723,
    0.00874074849817488,
    0.007939004566934372,
    0.007572991251945496,
    0.008008721073468526,
    0.008226583600044251,
    0.00813072317176395,
    0.0077124225431018404,
    0.007407412065400018,
    0.00753813087940216,
    0.007651419970724317,
    0.00788671460416582,
    0.00787799961037106,
    0.007459699511528015,
    0.007834428879949782,
    0.007764711942937638,
    0.008662316468026903,
    0.010204802287949457,
    0.011102408236927456,
    0.0119041512409846,
    0.012000010212262472,
    0.01119826528761122,
    0.010387807753350999,
    0.01058824168311225,
    0.00940305405192905,
    0.009525061183505587,
    0.009830071263843112,
    0.008775604565938315,
    0.009167762200037639,
    0.008087151447931925,
    0.008444450563854641,
    0.008854037589497037,
    0.009664496580759685,
    0.010901969803704156,
    0.011067546208699543,
    0.010753819942474365,
    0.009472776452700298,
    0.009124190078841316,
    0.008705889450179206,
    0.009176478054788377,
    0.008801750540733337,
    0.008758176962534588,
    0.008697173827224307,
    0.009193906585375467,
    0.01068410376707713,
    0.011337698830498592,
    0.01037909123632643,
    0.007755996684233347,
    0.006788675685723622,
    0.006239655299319161,
    0.006893250164058474,
    0.007677565481927659,
    0.008296302623218961,
    0.009647066858079699,
    0.009925932751761542,
    0.011215695142745973,
    0.011625281439887154,
    0.012505455546908907,
    0.012130728430218167,
    0.012993472682105172,
    0.013141621748606364,
    0.013673210806316801,
    0.014422665039698282,
    0.015721142954296535,
    0.017455348968505858,
    0.020165592034657797,
    0.01874511374367608,
    0.020967338614993627,
    0.02092376033465068,
    0.026466248830159505,
    0.028305029604170057,
    0.031965161561965945,
    0.029943370752864417,
    0.02978651636176639,
    0.03051853398482005,
    0.03298476523823208,
    0.036671025620566476,
    0.04034858730104235,
    0.039747274319330846,
    0.042074082295099895,
    0.042875825961430865,
    0.04572551396158007,
    0.04568193541632758,
    0.0479303134812249,
    0.047808316151301065,
    0.05173858867751228,
    0.053777796162499315,
    0.06027014209164513,
    0.05962526222070058,
    0.0648540114197466,
    0.06301523461937904,
    0.06459257285628055,
    0.06708495260940657,
    0.06901957968870799,
    0.07255768789185418,
    0.07761218229929606,
    0.07800435145696004,
    0.08119393640094334,
    0.08013945486810473,
    0.0801917568180296,
    0.07960787342654334,
    0.08067107021808624,
    0.08181269082758162,
    0.08539441128571829,
    0.08740749571058486,
    0.0907887593905131,
    0.09135521385404799,
    0.09357742402288649,
    0.09327241275045607,
    0.09505015307002597,
    0.09435299025641547,
    0.09618302583694459,
    0.09619173275099861,
    0.09928539858924017,
    0.10262307087580363,
    0.11002173741658529,
    0.11311540179782444,
    0.11932886918385825,
    0.1180478188726637,
    0.120287477572759,
    0.11996504081620112,
    0.11993894471062554,
    0.12140299240748087,
    0.12344220108456083,
    0.12506311469607884,
    0.12768619590335423,
    0.12795633686913385,
    0.1292635276582506,
    0.13228748798370363,
    0.13716765456729466,
    0.14238767623901366,
    0.14854891671074763,
    0.15338552792867025,
    0.15586055437723798,
    0.15850982348124185,
    0.15809153980678983,
    0.16162099520365397,
    0.16304140620761448,
    0.16866240395439994,
    0.17212214787801106,
    0.17682810571458604,
    0.17704586982727052,
    0.18061022758483888,
    0.1797037304772271,
    0.18371248457166886,
    0.18440962314605713,
    0.18658827251858182,
    0.185089381535848,
    0.18752947807312012,
    0.18945541063944496,
    0.1963748582204183,
    0.20020052274068198,
    0.20396530363294815,
    0.20455789777967667,
    0.2068672858344184,
    0.20787817478179935,
    0.21059717390272356,
    0.21069304360283744,
    0.2136473242441813,
    0.21068422317504884,
    0.21518099043104386,
    0.21249666849772134,
    0.21792583995395234,
    0.2145793925391303,
    0.21934625413682726,
    0.21114578353034127,
    0.2160433684455024,
    0.21131131119198268,
    0.22176884015401208,
    0.2232590240902371,
    0.2338036542468601,
    0.23067546049753826,
    0.23776917033725317,
    0.2337953315840827,
    0.24321577231089275,
    0.24059272925058997,
    0.24934212976031833,
    0.2479390589396159,
    0.25959038151635067,
    0.2590238881111145,
    0.2664487054612901,
    0.2620652514033847,
    0.26686699443393286,
    0.26273610803816055,
    0.26873177316453717,
    0.2653330394956801,
    0.2714159189330207,
    0.2675205712848239,
    0.27426565753089055,
    0.2751806338628133,
    0.28237883991665313,
    0.2832415866851807,
    0.28906292385525173,
    0.28521099567413327,
    0.29204319424099395,
    0.2901175324122111,
    0.2979433970981174,
    0.2992771291732788,
    0.3025799613528781,
    0.30175210846794975,
    0.3028763177659776,
    0.30036650127834746,
    0.3000614823235406,
    0.2982662826114231,
    0.2987804349263509,
    0.29749066140916613,
    0.2981268241670397,
    0.29517254723442926,
    0.2971333376566569,
    0.2961921339564853,
    0.2992509402169122,
    0.29991326385074196,
    0.30179560634824965,
    0.3029546263482836,
    0.3053250000211927,
    0.30501125812530516,
    0.3067018773820665,
    0.30562127113342286,
    0.30549926651848686,
    0.3063271453645494,
    0.30556898011101613,
    0.3042356766594781,
    0.2991027885013156,
    0.2955124028523763,
    0.28963007397121854,
    0.2903970040215386,
    0.28637954499986434,
    0.2872335444556342,
    0.2829894161224365,
    0.28182132720947267,
    0.2766185347239176,
    0.2772893640730116,
    0.27366411103142635,
    0.2758428213331434,
    0.2717382738325331,
    0.2714681529998779,
    0.26823503600226506,
    0.2718168046739366,
    0.2708843453725179,
    0.2772286563449436,
    0.27312407917446563,
    0.27701925913492836,
    0.27461402893066406,
    0.2828756607903375,
    0.2845575661129422,
    0.30074053022596575,
    0.3037731774648031,
    0.32245770348442926,
    0.3257169246673584,
    0.3456558322906494,
    0.34817435794406476,
    0.3611153390672472,
    0.36257067362467454,
    0.3739259931776258,
    0.3748758655124241,
    0.38304985894097215,
    0.38108903461032445,
    0.385062616136339,
    0.3793545956081814,
    0.3809059672885471,
    0.3753546439276801,
    0.37817868762546114,
    0.3742570241292318,
    0.3773597653706869,
    0.3730721452501085,
    0.37449275546603733,
    0.36941215727064347,
    0.37046673668755425,
    0.36612689759996203,
    0.36627506468031146,
    0.35996568891737196,
    0.357708617316352,
    0.3508240254720052,
    0.34973464330037435,
    0.34625750011867945,
    0.3459699291653104,
    0.33904182646009656,
    0.33365614997016063,
    0.3253074582417806,
    0.3201222356160482,
    0.3155642933315701,
    0.3125750944349501,
    0.3110150888231065,
    0.31168614281548396,
    0.31150312847561307,
    0.313089173634847,
    0.31282772064208986,
    0.31268829345703125,
    0.31056188371446397,
    0.3103876876831055,
    0.30875801510281037,
    0.30793023427327476,
    0.30846181657579214,
    0.3090980381435818,
    0.3096993086073134,
    0.31105865266588,
    0.31017843882242835,
    0.3106839095221625,
    0.30881891250610355,
    0.30998666551378035,
    0.3084790378146702,
    0.30738966200086804,
    0.3049495760599772,
    0.30183853149414064,
    0.2979692628648546,
    0.2944834603203667,
    0.2919039429558648,
    0.2886969926622179,
    0.2865532133314344,
    0.2844181866115994,
    0.28240513695610897,
    0.2808626619974772,
    0.27920688417222767,
    0.2787711673312717,
    0.2787275907728407,
    0.27885830561319985,
    0.2790848837958442,
    0.277010809580485,
    0.27609573152330186,
    0.2759997982449001,
    0.2767840915256076,
    0.2796512307061089,
    0.2812808566623264,
    0.28181243896484376,
    0.28278850131564676,
    0.2808799574110243,
    0.28165555318196617,
    0.2803483581542969,
    0.2807318030463325,
    0.28052263471815325,
    0.27938976287841794,
    0.2791370434231228,
    0.2794594785902235,
    0.28040935940212675,
    0.27957277086046006,
    0.27592137230767144,
    0.27586908976236985,
    0.2717296557956272,
    0.27151179843478734,
    0.2691240056355794,
    0.26888000064426,
    0.2665009307861328,
    0.2689148712158203,
    0.2688626437717014,
    0.26946395026312936,
    0.26784302181667746,
    0.2672417153252496,
    0.2655772484673394,
    0.26767746395534936,
    0.26973410288492833,
    0.2736992475721571,
    0.27598252614339197,
    0.2789890586005317,
    0.27899781545003255,
    0.28042701297336153,
    0.27775166829427084,
    0.2820828247070312,
    0.28372987959120005,
    0.288845329284668,
    0.2918867195977105,
    0.29410892486572265,
    0.2954335488213433,
    0.29741173214382594,
    0.29984308030870227,
    0.3022134314643012,
    0.3039650599161784,
    0.30549881829155817,
    0.30626568688286676,
    0.30678855895996093,
    0.3053681055704753,
    0.3056643803914388,
    0.3043223614162869,
    0.30298901875813805,
    0.30060125562879775,
    0.29665357377794055,
    0.293943362765842,
    0.2923573260837131,
    0.2919041951497396,
    0.2907625749376085,
    0.29044015248616534,
    0.2875295045640734,
    0.28746852027045355,
    0.284043706258138,
    0.2812898847791883,
    0.27844896104600697,
    0.27383898841010196,
    0.271825926038954,
    0.2701440217759874,
    0.26933356391059027,
    0.2670416344536675,
    0.26551659478081596,
    0.26359066009521487,
    0.26373009575737844,
    0.2645928446451823,
    0.26659719255235464,
    0.26752965715196403,
    0.26813096788194446,
    0.2686625205145942,
    0.26915927886962887,
    0.26796536763509116,
    0.2681135347154405,
    0.26586516910129127,
    0.2659610133700901,
    0.2663357586330838,
    0.2666494878133138,
    0.26900241427951394,
    0.2667976294623481,
    0.2649326833089192,
    0.2615601433648003,
    0.2575601577758789,
    0.2553989664713542,
    0.25144253624810115,
    0.2488891643948025,
    0.24686730278862848,
    0.243451173570421,
    0.24289336734347874,
    0.2391983328925239,
    0.23500657823350693,
    0.22967318216959634,
    0.2260217963324653,
    0.22466230604383683,
    0.223912845187717,
    0.2245577324761285,
    0.2236775588989258,
    0.2222745132446289,
    0.21899784935845268,
    0.21945972866482202,
    0.21755993949042426,
    0.21691505432128907,
    0.21555558522542317,
    0.21414381451076933,
    0.21477996614244246,
    0.21519826889038085,
    0.21858826107449003,
    0.21914596981472437,
    0.22061876085069443,
    0.22162093692355686,
    0.22196949005126954,
    0.22258822123209634,
    0.2219782002766927,
    0.22240520477294923,
    0.22251850552029082,
    0.22350325690375436,
    0.22626576317681205,
    0.2285141075981988,
    0.23296725379096136,
    0.23262740241156687,
    0.235294066535102,
    0.23110233306884767,
    0.23092804378933374,
    0.2286535178290473,
    0.2296382988823785,
    0.23260996924506294,
    0.23542476230197484,
    0.23567747751871745,
    0.23469278547498917,
    0.23218296898735893,
    0.23369059244791668,
    0.23574725257025822,
    0.24094994015163848,
    0.24100221845838757,
    0.24264060126410592,
    0.24001750098334418,
    0.23665367126464842,
    0.2357996368408203,
    0.23289768642849393,
    0.2307016118367513,
    0.23002191331651475,
    0.22816571129692922,
    0.22847073449028865,
    0.22723325941297742,
    0.22653611077202693,
    0.2246363491482205,
    0.22361673143174912,
    0.2229108640882704,
    0.22199580722384982,
    0.22057533688015407,
    0.21837925804985894,
    0.2186319563123915,
    0.21886726803249787,
    0.21913743336995442,
    0.21941634707980684,
    0.21763854556613496,
    0.21521588643391928,
    0.21397840711805557,
    0.21189562903510198,
    0.21163419935438368,
    0.21125074598524304,
    0.2112071948581272,
    0.21224424574110246,
    0.21231394873725043,
    0.21331613752577042,
    0.2129762691921658,
    0.21232267167833116,
    0.21051873948838976,
    0.2081135008070204,
    0.2052637905544705,
    0.20346856011284722,
    0.20270166609022353,
    0.20213523864746094,
    0.2031897015041775,
    0.20250127156575518,
    0.20212653266059025,
    0.2008977762858073,
    0.1995731692843967,
    0.19631386227077907,
    0.19200880686442057,
    0.18755552927652996,
    0.18296289232042098,
    0.17933760748969185,
    0.1733854971991645,
    0.1679998779296875,
    0.16341601053873697,
    0.1596077389187283,
    0.1587101279364692,
    0.15628748151991104,
    0.15312408023410373,
    0.1511197280883789,
    0.14834849463568794,
    0.1461262681749132,
    0.14393018934461804,
    0.1416992653740777,
    0.1395467546251085,
    0.13762955983479816,
    0.13574721866183811,
    0.13328971862792968,
    0.13049235026041664,
    0.12921131134033204,
    0.12691939883761935,
    0.1248540539211697,
    0.12095866309271919,
    0.11617435031467012,
    0.11283668729994033,
    0.10989985783894857,
    0.10962098651462131,
    0.11027459886338975,
    0.11028331544664172,
    0.1090807024637858,
    0.10587373521592881,
    0.10213515811496311,
    0.09880615658230252,
    0.09730724546644422,
    0.09623534520467121,
    0.09762097888522679,
    0.09670593473646376,
    0.09560787836710612,
    0.0928453403049045,
    0.09173860549926759,
    0.08954250971476237,
    0.0883224572075738,
    0.08795645183987089,
    0.0867886839972602,
    0.08787800470987955,
    0.0862222311231825,
    0.08484532250298395,
    0.08261437733968098,
    0.07950325859917534,
    0.07721132490370008,
    0.07629628923204211,
    0.07505880567762586,
    0.07279301537407769,
    0.06941176096598307,
    0.06462745454576281,
    0.06332027223375108,
    0.06049674246046278,
    0.058980403476291236,
    0.05719391610887315,
    0.05385623931884765,
    0.05240090158250597,
    0.05193030251397027,
    0.051930299335055885,
    0.05132899814181857,
    0.048191744486490884,
    0.04540307998657226,
    0.04264055040147569,
    0.03984316190083822,
    0.03850982666015625,
    0.03417867130703397,
    0.032514181137084965,
    0.03186058150397406,
    0.02992594083150228,
    0.02957735856374105,
    0.028592608239915634,
    0.03044010268317329,
    0.030945555369059245,
    0.030745111571417915,
    0.028549031681484646,
    0.025690653059217663,
    0.025629647572835285,
    0.02738128079308404,
    0.027973878648546006,
    0.027834439277648924,
    0.026248382992214627,
    0.02387801700168186,
    0.021716793908013236,
    0.01938999573389689,
    0.017917226950327558,
    0.018161233133739896,
    0.01798694199985928,
    0.0203050246503618,
    0.01953814148902893,
    0.01871025330490536,
    0.016801757083998784,
    0.013830076356728872,
    0.013795218235916562,
]

# Main Driver Code

# Functionality


def plot_spectrum_with_regression(
    x_values, y_values, title, xlabel, ylabel, saveFilename
):

    # Create Scatter Plot
    plt.scatter(x_values, y_values, color="blue", label="Data Points")

    # Reshape x_values for linear regression ( it expects a 2D array )
    x_values_reshaped = np.array(x_values).reshape(-1, 1)

    # perform linear regression
    reg = LinearRegression().fit(x_values_reshaped, y_values)

    # predict y values based on regression model
    y_pred = reg.predict(x_values_reshaped)

    # plot the regression line
    plt.plot(x_values, y_pred, color="red", label="Best Fit Line")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.xlim(min(x_values), max(x_values))
    plt.ylim(min(y_values) - 0.1, max(y_values) + 0.1)

    plt.savefig(saveFilename)
    plt.show()


def absorbance_spectrun_scatter():
    global saveFilename

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the absorbance spectrum and normalize
    absorbances = normalise(absorbance(reference, spectrum))

    # Get wavelength axis using polynomial fit
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = np.polyval(params, range(len(spectrum)))

    # Plot absorbance with regression line
    plot_spectrum_with_regression(
        nmAxis,
        absorbances,
        "Absorbance Spectrum",
        "Wavelength (nm)",
        "Absorbance",
        saveFilename,
    )


def absorbance_spectrun():

    global saveFilename

    # Plot Parameters

    patchLabels = ["Intensity"]
    plotColors = ["black"]

    title = "Absorbance Spectrum"

    # Capture The Reference Spectrum

    spectrum = capture()

    # Getting The Absorbance Spectrum

    absorbances = []
    absorbances.append(normalise(absorbance(reference, spectrum)))

    # Fiding Out The Cofficients Of The Polynomial

    params = np.polyfit(pixel, wavelength, 3)

    # return p = np.poly1d(range) - This Returns The Polynomial

    # Solving The Polynomial For Every Pixel
    # Assigning To Every Pixel Its Corresponding Wavelength

    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # NOTE: This Operation Is Not Needed As The Polynomial Is Already A Function

    # nmAxis = np.poly1d(len(spectrum))
    # print(len(patchLabels), len(spectraToPlot))

    if len(patchLabels) < len(absorbances):
        patchLabels.append([""] * (len(absorbances) - len(patchLabels)))

    if len(plotColors) < len(absorbances):
        plotColors = []
        for i in range(len(absorbances)):
            plotColors.append(colors[i])

    patches = []
    for i in range(len(patchLabels)):
        patches.append(mpatches.Patch(color=plotColors[i], label=patchLabels[i]))

    colorCounter = 0
    for spectrum in absorbances:
        plt.plot(nmAxis, spectrum, color=plotColors[colorCounter])
        colorCounter += 1

    plt.title(title)
    plt.legend(handles=patches)
    plt.xlim(300, 800)
    plt.ylim(0, 5)
    plt.xlabel("Wavelegth (nm)")
    plt.ylabel("Absorbance")

    plt.savefig(saveFilename)
    plt.show()


def transmittance_spectrun_scatter():
    global saveFilename

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the transmittance spectrum and normalize
    transmittances = normalise(transmittance(reference, spectrum))

    # Get wavelength axis using polynomial fit
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = np.polyval(params, range(len(spectrum)))

    # Plot transmittance with regression line
    plot_spectrum_with_regression(
        nmAxis,
        transmittances,
        "Transmittance Spectrum",
        "Wavelength (nm)",
        "Transmittance",
        saveFilename,
    )


def transmittance_spectrun():

    global saveFilename

    # Plot Parameters

    patchLabels = ["Intensity"]
    plotColors = ["black"]

    title = "Transmittance Spectrum"

    # Capture The Reference Spectrum

    spectrum = capture()

    transmittances = []
    transmittances.append(normalise(transmittance(reference, spectrum)))

    # Fiding Out The Cofficients Of The Polynomial

    params = np.polyfit(pixel, wavelength, 3)

    # return p = np.poly1d(range) - This Returns The Polynomial

    # Solving The Polynomial For Every Pixel
    # Assigning To Every Pixel Its Corresponding Wavelength

    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # NOTE: This Operation Is Not Needed As The Polynomial Is Already A Function

    # nmAxis = np.poly1d(len(spectrum))
    # print(len(patchLabels), len(spectraToPlot))

    if len(patchLabels) < len(transmittances):
        patchLabels.append([""] * (len(transmittances) - len(patchLabels)))

    if len(plotColors) < len(transmittances):
        plotColors = []
        for i in range(len(transmittances)):
            plotColors.append(colors[i])

    patches = []
    for i in range(len(patchLabels)):
        patches.append(mpatches.Patch(color=plotColors[i], label=patchLabels[i]))

    colorCounter = 0
    for spectrum in transmittances:
        plt.plot(nmAxis, spectrum, color=plotColors[colorCounter])
        colorCounter += 1

    plt.title(title)
    plt.legend(handles=patches)
    plt.xlim(300, 800)
    plt.ylim(0, 5)
    plt.xlabel("Wavelegth (nm)")
    plt.ylabel("Transmittance")

    plt.savefig(saveFilename)
    plt.show()


def reflectance_spectrun_scatter():
    global saveFilename

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the reflectance spectrum and normalize
    reflectances = normalise(reflectance(reference, spectrum))

    # Get wavelength axis using polynomial fit
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = np.polyval(params, range(len(spectrum)))

    # Plot reflectance with regression line
    plot_spectrum_with_regression(
        nmAxis,
        reflectances,
        "Reflectance Spectrum",
        "Wavelength (nm)",
        "Reflectance",
        saveFilename,
    )


def reflectance_spectrun():

    global saveFilename

    # Plot Parameters

    patchLabels = ["Intensity"]
    plotColors = ["black"]

    title = "Reflectance Spectrum"

    # Capture The Reference Spectrum

    spectrum = capture()

    reflectances = []
    reflectances.append(normalise(reflectance(reference, spectrum)))

    # Fiding Out The Cofficients Of The Polynomial

    params = np.polyfit(pixel, wavelength, 3)

    # return p = np.poly1d(range) - This Returns The Polynomial

    # Solving The Polynomial For Every Pixel
    # Assigning To Every Pixel Its Corresponding Wavelength

    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # NOTE: This Operation Is Not Needed As The Polynomial Is Already A Function

    # nmAxis = np.poly1d(len(spectrum))
    # print(len(patchLabels), len(spectraToPlot))

    if len(patchLabels) < len(reflectances):
        patchLabels.append([""] * (len(reflectances) - len(patchLabels)))

    if len(plotColors) < len(reflectances):
        plotColors = []
        for i in range(len(reflectances)):
            plotColors.append(colors[i])

    patches = []
    for i in range(len(patchLabels)):
        patches.append(mpatches.Patch(color=plotColors[i], label=patchLabels[i]))

    colorCounter = 0
    for spectrum in reflectances:
        plt.plot(nmAxis, spectrum, color=plotColors[colorCounter])
        colorCounter += 1

    plt.title(title)
    plt.legend(handles=patches)
    plt.xlim(300, 800)
    plt.ylim(0, 5)
    plt.xlabel("Wavelegth (nm)")
    plt.ylabel("Reflectance")

    plt.savefig(saveFilename)
    plt.show()


def scatter_reflectance_spectrum():

    global saveFilename

    # Plot Parameters
    patchLabels = ["Intensity"]
    plotColors = ["black"]
    title = "Reflectance Spectrum"

    # Capture The Reference Spectrum
    spectrum = capture()

    reflectances = []
    reflectances.append(normalise(reflectance(reference, spectrum)))

    # Finding Out The Coefficients Of The Polynomial
    params = np.polyfit(pixel, wavelength, 3)

    # Solving The Polynomial For Every Pixel
    nmAxis = np.polyval(params, np.arange(len(spectrum)))

    # Ensure patchLabels and plotColors are aligned with reflectances
    if len(patchLabels) < len(reflectances):
        patchLabels.append([""] * (len(reflectances) - len(patchLabels)))

    if len(plotColors) < len(reflectances):
        plotColors = []
        for i in range(len(reflectances)):
            plotColors.append(colors[i])

    patches = []
    for i in range(len(patchLabels)):
        patches.append(mpatches.Patch(color=plotColors[i], label=patchLabels[i]))

    colorCounter = 0
    for spectrum in reflectances:
        plt.scatter(
            nmAxis,
            spectrum,
            color=plotColors[colorCounter],
            label=patchLabels[colorCounter],
        )
        colorCounter += 1

    plt.title(title)
    plt.legend(handles=patches)
    plt.xlim(300, 800)
    plt.ylim(0, 5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")

    plt.savefig(saveFilename)
    plt.show()


def scikit_scatter_plot():
    i_dist = capture()

    if i_dist is not None:
        # Preprocess data with Scikit-Learn
        scaler = StandardScaler()
        i_dist_scaled = scaler.fit_transform(i_dist)

        # Generate nmAxis for plotting
        nmAxis = np.arange(len(i_dist_scaled))

        # Create scatter plot
        plt.scatter(nmAxis, i_dist_scaled, color="black", label="Reflectance Spectrum")
        plt.title("Reflectance Spectrum")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Reflectance")
        plt.xlim(0, len(i_dist_scaled))
        plt.ylim(-2, 2)  # Adjust limits based on data scaling
        plt.legend()
        plt.savefig("scatter_plot_test.png")
        plt.show()
    else:
        print("No data to plot.")


def apply_kmeans_clustering(spectrum, n_clusters=3):
    # Convert the spectrum data to a 2D array (n_samples x n_features)
    spectrum_reshaped = np.array(spectrum).reshape(-1, 1)

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the model to the spectrum data
    kmeans.fit(spectrum_reshaped)

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    return cluster_labels, kmeans.cluster_centers_


def plot_kmeans_clusters(wavelengths, spectrum, cluster_labels, centers):
    # Scatter plot of the data points, colored by their cluster labels
    plt.scatter(
        wavelengths, spectrum, c=cluster_labels, cmap="viridis", label="Data Points"
    )

    # Plot the cluster centers
    plt.scatter(
        wavelengths,
        centers[cluster_labels],
        color="red",
        marker="x",
        label="Cluster Centers",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral Intensity")
    plt.title("KMeans Clustering on Spectral Data")
    plt.legend()
    plt.show()


def elbow_method():

    spectrum = capture()

    spectrum_reshaped = np.array(spectrum).reshape(-1, 1)
    wcss = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(spectrum_reshaped)
        wcss.append(
            kmeans.inertia_
        )  # Sum of squared distances to the nearest cluster center

    plt.plot(range(1, 10), wcss, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.show()


def apply_regression_per_cluster(wavelengths, spectrum, cluster_labels):
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)
        wavelengths_cluster = np.array(wavelengths)[cluster_indices]
        spectrum_cluster = np.array(spectrum)[cluster_indices]

        # Apply linear regression to each cluster
        reg = LinearRegression()
        reg.fit(wavelengths_cluster.reshape(-1, 1), spectrum_cluster)
        y_pred = reg.predict(wavelengths_cluster.reshape(-1, 1))

        # Plot regression line for each cluster
        plt.scatter(wavelengths_cluster, spectrum_cluster, label=f"Cluster {cluster}")
        plt.plot(wavelengths_cluster, y_pred, label=f"Regression Cluster {cluster}")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral Intensity")
    plt.legend()
    plt.show()


def kmeans_absorbance_spectrun(n_clusters):
    global saveFilename

    # Plot Parameters
    title = "KMeans Clustering on Absorbance Spectrum"

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the absorbance using your reference data
    absorbances = absorbance(reference, spectrum)

    # Normalize the absorbance data
    absorbances_normalized = normalise(absorbances)

    # Prepare the data for KMeans clustering
    absorbances_reshaped = np.array(absorbances_normalized).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(absorbances_reshaped)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Polynomial fitting for wavelength axis (assuming 'pixel' and 'wavelength' are defined)
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # Plot the clustered data
    plt.scatter(
        nmAxis,
        absorbances_normalized,
        c=cluster_labels,
        cmap="viridis",
        label="Data Points",
    )

    # Plot cluster centers
    for i, center in enumerate(centers):
        plt.scatter(
            nmAxis,
            [center] * len(nmAxis),
            color="red",
            marker="x",
            label=f"Cluster Center {i+1}",
        )

    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Absorbance")
    plt.legend()
    plt.xlim(300, 800)
    plt.ylim(0, 1)
    plt.show()


def reflectance_kmeans_clustering(reference, wavelength):
    # Calculate reflectance spectrum

    sample = capture()

    reflectance_spectrum = reflectance(reference, sample)

    # Reshape reflectance data for KMeans (1 feature per sample)
    reflectance_spectrum = np.array(reflectance_spectrum).reshape(-1, 1)

    # Standardize the reflectance spectrum for better clustering
    scaler = StandardScaler()
    reflectance_scaled = scaler.fit_transform(reflectance_spectrum)

    # Apply KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(reflectance_scaled)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Scatter plot of reflectance spectrum with cluster coloring
    plt.scatter(
        wavelength, reflectance_spectrum, c=cluster_labels, cmap="viridis", s=100
    )
    plt.title("Reflectance Spectrum KMeans Clustering")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.colorbar(label="Cluster")
    plt.show()


def kmeans_transmittance_spectrun(n_clusters):
    global saveFilename

    # Plot Parameters
    title = "KMeans Clustering on Transmittance Spectrum"

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the transmittance using your reference data
    transmittances = transmittance(reference, spectrum)

    # Normalize the transmittance data
    transmittances_normalized = normalise(transmittances)

    # Prepare the data for KMeans clustering
    transmittances_reshaped = np.array(transmittances_normalized).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(transmittances_reshaped)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Polynomial fitting for wavelength axis (assuming 'pixel' and 'wavelength' are defined)
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # Plot the clustered data
    plt.scatter(
        nmAxis,
        transmittances_normalized,
        c=cluster_labels,
        cmap="viridis",
        label="Data Points",
    )

    # Plot cluster centers
    for i, center in enumerate(centers):
        plt.scatter(
            nmAxis,
            [center] * len(nmAxis),
            color="red",
            marker="x",
            label=f"Cluster Center {i+1}",
        )

    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Transmittance")
    plt.legend()
    plt.xlim(300, 800)
    plt.ylim(0, 1)
    plt.show()

def kmeans_reflectance(n_clusters=3):
    global saveFilename

    # Plot Parameters
    title = "KMeans Clustering on Transmittance Spectrum"

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the transmittance using your reference data
    transmittances = reflectance(reference, spectrum)

    # Normalize the transmittance data
    transmittances_normalized = normalise(transmittances)

    # Prepare the data for KMeans clustering
    transmittances_reshaped = np.array(transmittances_normalized).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(transmittances_reshaped)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Polynomial fitting for wavelength axis (assuming 'pixel' and 'wavelength' are defined)
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # Plot the clustered data
    plt.scatter(
        nmAxis,
        transmittances_normalized,
        c=cluster_labels,
        cmap="viridis",
        label="Data Points",
    )

    # Plot cluster centers
    for i, center in enumerate(centers):
        plt.scatter(
            nmAxis,
            [center] * len(nmAxis),
            color="red",
            marker="x",
            label=f"Cluster Center {i+1}",
        )

    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Transmittance")
    plt.legend()
    plt.xlim(300, 800)
    plt.ylim(0, 1)
    plt.show()

def kmeans_absorbance(n_clusters=3):
    global saveFilename

    # Plot Parameters
    title = "KMeans Clustering on Transmittance Spectrum"

    # Capture The Reference Spectrum
    spectrum = capture()

    # Calculate the transmittance using your reference data
    transmittances = absorbance(reference, spectrum)

    # Normalize the transmittance data
    transmittances_normalized = normalise(transmittances)

    # Prepare the data for KMeans clustering
    transmittances_reshaped = np.array(transmittances_normalized).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(transmittances_reshaped)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Polynomial fitting for wavelength axis (assuming 'pixel' and 'wavelength' are defined)
    params = np.polyfit(pixel, wavelength, 3)
    nmAxis = []
    for i in range(len(spectrum)):
        v1 = params[0] * float(i**3)
        v2 = params[1] * float(i**2)
        v3 = params[2] * float(i**1)
        v4 = params[3] * float(i**0)
        nmAxis.append(v1 + v2 + v3 + v4)

    # Plot the clustered data
    plt.scatter(
        nmAxis,
        transmittances_normalized,
        c=cluster_labels,
        cmap="viridis",
        label="Data Points",
    )

    # Plot cluster centers
    for i, center in enumerate(centers):
        plt.scatter(
            nmAxis,
            [center] * len(nmAxis),
            color="red",
            marker="x",
            label=f"Cluster Center {i+1}",
        )

    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Transmittance")
    plt.legend()
    plt.xlim(300, 800)
    plt.ylim(0, 1)
    plt.show()
# Main Driver Code

# Defining The Folder Where The Results Will Be Saved

folder_name = "Results"

# Checking If The Folder Exists
if os.path.exists(folder_name):
    pass
else:
    os.mkdir(folder_name)


fname = input("Enter The Name Of The File : ")
saveFilename = os.path.join(folder_name, "{}.png".format(fname))

print()
print("Choose A Fnction To Perform")
print()

print("2. Transmittance Spectrum")
print("3. Reflectance Spectrum")
print("1. Absorbance Spectrum")
print("4. Scatter Plot Test")
print("5. SkLearn Plot")
print()
print("6. Scatter Transmittance Spectrum")
print("7. Scatter Reflectance Spectrum")
print("8. Scatter Absorbance Spectrum")
print()
print("9. Kmeans Transmittance Spectrum")
print("10. Kmeans Reflectance Spectrum")
print("11. Kmeans Absorbance Spectrum")
print()
print("12. Elbow Method")
print()
print("13. Kmeans Clustering For Samples")

print()

choice = int(input("Enter Your Choice : "))
print()

if choice == 1:
    absorbance_spectrun()

elif choice == 2:
    transmittance_spectrun()

elif choice == 3:
    reflectance_spectrun()

elif choice == 4:
    scatter_reflectance_spectrum()

elif choice == 5:
    scikit_scatter_plot()

elif choice == 6:
    transmittance_spectrun_scatter()

elif choice == 7:
    reflectance_spectrun_scatter()

elif choice == 8:
    absorbance_spectrun_scatter()

elif choice == 9:
    kmeans_transmittance_spectrun(3)

elif choice == 10:
    kmeans_reflectance(3)

elif choice == 11:
    kmeans_absorbance(3)

elif choice == 12:
    clusters = elbow_method()

else:
    print("Invalid Choice")
    exit()
