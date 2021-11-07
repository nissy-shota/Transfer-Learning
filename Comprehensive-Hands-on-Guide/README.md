# A Comprehensive Hands-on Guide to Transfer Learning

## Introduction

人間はタスク感で知識を伝達する能力がある．
それを機械学習でも実施しようという試み．
1つのタスクについて学習し．関連するタスクを解決するときも同じように活用する．
タスクの関連性があ高ければ高いほど，知識の伝達が容易になる．
たとえば
- Know how to ride a motorbike ⮫ Learn how to ride a car
- Know how to play classic piano ⮫ Learn how to play jazz piano
- Know math and statistics ⮫ Learn machine learning
が挙げられる．

新しい局面やトピックを学ぶ時，すべてをゼロから学ぶのではなく，過去に学んだことから知識を移して活用する．  
これまでの機械学習や深層学習のアルゴリズムは伝統的に単独で動作するように設計されてきた．  
これらのアルゴリズムは，特定のタスクを解決するように訓練されている．  
よって，特徴量空間の分布が変わるとモデルを位置から作り直すひつようがある．  
転移学習は，孤立した学習パラダイムを克服し，あるタスクで獲得した知識を関連するタスクの解決に活用するという考え方．  

転移学習の概念，範囲，応用について述べる．  

- Motivation for Transfer Learning
- Understanding Transfer Learning
- Transfer Learning Strategies
- Transfer Learning for Deep Learning
- Deep Transfer Learning Strategies
- Types of Deep Transfer Learning
- Applications of Transfer Learning
- Case Study 1: Image Classification with a Data Availability Constraint
- Case Study 2: Multi-Class Fine-grained Image Classification with Large Number of Classes and Less Data Availability
- Transfer Learning Advantages
- Transfer Learning Challenges
- Conclusion & Future Scope

機械学習や統計モデリングの時代から始まった一般的な上位概念としての伝達学習を見ていく．  
本稿では，深層学習を中心に説明していく．  

## Motivation for Transfer Learning

[Neural Information Processing Systems Tutorial](https://youtu.be/wjqaz6m42wU)  

転移学習は2010年代に入ってから登場した概念ではない．  
研究者や学術書によって定義が異なることが合ったが，GoodfellowらはDeep Learningのなかで，  
> Situation where what has been learned in one setting is exploited to improve generalization in another setting.  
と定義している．  
（訳：ある環境で学習されたことが別の環境での一般化を向上させるために利用される状況）  

複雑な問題を解決するモデルには大量のデータが必用であり，教師付きモデルのために大量のラベル付きデータを取得することは，データポイントのラベル付にかかる時間と労力を考えると，
非常に困難である．例えばあ，ImageNetはスタンフォード大学の長年の努力の結果様々なカテゴリーに属する何百万もの画像を持っている．  

しかし，すべてのドメインに対して，そのようなデータセットを入手するのは困難である．  
また，ほとんどの深層学習モデルは特定のドメイン，あるいは特定のタスクに非常に特化している．  
これらのモデルは，非常に高い精度を持ち，すべてのベンチマークを打ち負かす最先端のモデルであるが，訓練されたタスクと類似しているかもしれない，  
新しいタスクに使用した場合は，パフォーマンスが大幅に低下してしまう．   
これが特定のタスクやドメインを超えて事前に学習されたモデルから得られた知識を活用し，新たな問題を解決する方法を見出そうとする点が転移学習の動機である．  

## Understanding Transfer Learning

転移学習は真祖具悪臭に特化した新しい概念ではないということ．  
機械学習モデルを構築して，トレーニングする従来のアプローチと点学習の原理に従った方法論を用いるには大きな違いがある．  

従来の学習は独立しており，特定のタスクやデータセットに基づいて純粋に学習し，その上で個別の孤立したモデルを学習する．  
そのため，あるモデルから別のモデルに移行できる知識は保存されない．  
転移学習では以前に学習したモデルの知識，  
特徴や重みなどを新しいモデルの学習に活用することができる．  
そのため，新しいタスクのためのデータが少ないなどの問題にも対処することができる．  

従来のMLアルゴリズムは，与えられたドメイン内の必用なタスクに対して十分な訓練例がない場合破綻する．  
モデルが学習データや領域に偏っていると表現される．  

転移学習では，タスクT1で学習したデータが大幅に多ければ，T1で学習した特徴や重みを新たなタスクT2へと適用することができる．  
Computer Visionの領域では，エッジ、形状、コーナー、強度などの低レベルの特徴をタスク間で共有することで、タスク間の知識伝達が可能になる．  

## Formal Definition

ここでは，転移学習の正式な定義を見てから様々な戦略を理解するために活用してみる，．

> A domain, D, is defined as a two-element tuple consisting of feature space, χ, and marginal probability, P(Χ), where Χ is a sample data point. Thus, we can represent the domain mathematically as D = {χ, P(Χ)}

![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/1.png)

ここで，xiは特定のベクトルを表す．一報タスクTはラベル空間ｙと目的関数ηの2要素タプルとして定義される．
また目的関数は確率的な観点からPと表すことができる．

![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/2.png)
![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/3.png)
![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/4.png)
![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/5.png)

転移学習とはソース学習器の知識をターゲとタスクに活用する能力を持つことである．
- What to transfer  
  何を転移するか．ターゲットタスクのパフォーマンスを向上させるために，ソースからターゲットに知識のどの部分を転送することができるかという答えを求めようとする．
  知識の土の部分がソース特有のもので，何がソースとターゲットの間で共通しているかを特定する．
- When to transfer  
  目的のために知識を転移すると何も改善されないどころか，自体が悪化する場合もある．negative transfer.
  目標とするタスクのパフォーマンスを向上させるために転移学習を利用することを目指すべきであり，劣化させるべきではない．
  いつ転送していつ転送しないかを慎重に判断する必用がある．
- How to transfer  
  what, whenの答えがでたら，次は知識をドメインとタスクの間で実際に転移する方法のとく手に進みます．  
  そのためには既存のアルゴリズムに変更を加えたり，様々なテクニックを駆使したりする必用がありますが，これについては本記事の最後で述べる．
  
## Transfer Learning Strategies

転移学習には様々な戦略やテクニックがあり，ドメインやタスク，データの有無などに応じて適用する．  
転移学習に関する論文![A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
を参照すると良い．  

![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/6.png)  



上記の図に基づいて転移学習の方法は，次のような従来のMLアルゴリズムのタイプに基づいて分類できる．  

- Inductive Transfer learning  
  ソースドメインとターゲットドメインは同じ.  
  ソースタスクとドメインタスクはお互いに異なる．  
  アルゴリズムはソースドメインの帰納的バイアスを利用して，ターゲットタスクの改善に役立てようとする．  
  これは更に２つのサブカテゴリーに分けられ，それぞれマルチタスク学習と自己学習に似ている．  
  
- Unsupervised Transfer Learning   
  Inductive Transfer learningに似ている．ターゲットドメインにおける教師なしのタスクに焦点を当てている．  
  ソースドメインとターゲットドメインは似ているがタスクは異なる．  
  このシナリオではどちらかのドメインでラベル付きデータが利用できない．  
  
- Transductive Transfer Learning  
  このシナリオは，ソースタスクとターゲットタスクの間に類似性があるが，対応するドメインは異なる．  
  この設定では，ソースドメインには多くのラベル付きデータがあるが，ターゲットドメインにはラベルがない．  
  更に特徴空間が異なる場合や，marginal probabiliteiesが異なる場合に個々に分類される．  

We can summarize the different settings and scenarios for each of the above techniques in the following table.  


![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/8.png)  

3つのtransfer categoriesはtransfer learningが適用され．研究できる設定の概要を示している．  
これらのカテゴリー感で何をtransferするかというアプローチを以下に示す．  

- Instance transfer  

  ソースドメインの知識をターゲットタスクに再利用する．もっとも理想的なシナリオ．
  ソースドメインのデータを直接再利用することはできない．  
  ソースドメインから特定のインスタンスがあり，それをターゲットデータと一緒に再利用することで，結果を改善することができる．  
- Feature-representation transfer  
  この手法はソースドメインからターゲットドメインに利用できる優れた特徴表現を特定することで，ドメインの分岐を最小限に抑えエラー率を低減する．  
  特徴表現に基づくtransferにはラベル付きデータの有無に応じて教師あり，なしを適用できる．  
- Parameter transfer  
  関連するタスクのモデルがいくつかのパラメタやハイパーパラメタの事前分布を共有しているという前提が必用．  
  ソースタスクとターゲットタスクの療法を同時に学習するマルチタスクとは異なる．  
  transfer learningでは，全体のパフォーマンスを向上させるために，ターゲットドメインの損失に追加の重みを適用することもある．  
- Relational-knowledge transfer  
  前述のアプローチとは異なり，独立同一分布ではないデータや非IDデータを扱おうとするもの．  
  核で０他ポイントが他のデータポイントと関係を持っているようなデータ．  
  ソーシャルネットワークのデータに利用する．  

The following table clearly summarizes the relationship between different transfer learning strategies and what to transfer.  


![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/7.png)  


transfer learning stategies and what to transfer summarizes  


## Transfer Learning for Deep Learning

全セクションでは機械学習に適用できる一般的なアプローチ．
深層学習の文脈で転移学習を適用できるのかという疑問が生じる．

![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/9.png)  

transfer learning in DL

深層学習のモデルは帰納的学習と呼ばれるものの代表である．  
帰納的学習のアルゴリズムの目的は学習例のセットからマッピングを推論することである．  
例えば，分類の場合，モデルは入力特徴とクラスラベルの間のマッピングを学習する．  
このような学習機がunseenのデータに対してうまく一般化するためにはそのアルゴリズムは学習データの分布に関連する一連の過程を用いて動作する．  
この過程を帰納的バイアスと呼ぶ．帰納的バイアスや過程は制限する仮説空間や仮説空間の探索プロセスなど，複数の要因によって特徴づけられる．  
これらのバイアスは与えられたタスクやドメインに置いてモデルが何をどのように学習するかに影響を与える．  


![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/10.png)  

transfer learning idea  

inductive transfer learning techniquesはソースタスクの帰納的バイアスをターゲットタスクの支援に利用する．  
これには，モデルの空間を限定してターゲとタスクの帰納的バイアスを調整したり，仮説空間を絞り込んだり，ソースタスクの知識を利用して探索プロセス自体を調整するなど様々な方法がある．  


![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/11.png)  

inductive transfer

iductive transferとは別に帰納的学習アルゴリズムはベイジアン及び階層的transfer手法を利用してターゲットタスクの学習とパフォーマンスの向上を支援．


## Deep Transfer Learning Strategies

深層学習は大きな進歩を挙げているが，必用な学習時間とデータ量は従来のMLシステムに比べて遥かに多い．  
CVやNLPなどの領域で最先端の性能をもつ様々な深層学習ネットワークが開発され，テストされている．  
これらの事前学習されたモデルやネットワークは深層学習における転移学習の基礎となり，deep transfer learningと呼ばれる．  

### Off-the-shelf Pre-trained Models as Feature Extractors

深層学習システムやモデルは異なる階層で（Layerの特徴の階層的な表現）で異なる特徴を学習するLayerをもつアーキテクチャである．  
最終的に最後のLayer（教室器学習の場合通常はFully connected layer）に接続され，最終的な出力が得られる．  
このレイヤー構造により，最終レイヤーを持たない事前学習済みのネットワークInception V3やVGGなどを他のタスクの固定的な特徴量抽出器として活用することができる．  

**ここでの重要なアイデアは、事前に学習されたモデルの重み付けされた層を利用して特徴を抽出するだけで、新しいタスクのための新しいデータでの学習中にモデルの層の重みを更新しないということです。**  

たとえば，最終分類層を持たないAlexNetを利用すると新しいドメインタスクからの画像を隠れ状態に基づいて4096次元のベクトルに変換し，ソースドメインタスクの知識を利用して，  
新しいドメインタスクから特徴を抽出できる．  
これはdeep neural networkを用いたtransfer learningを行う上で最も利用されている手法の一つ．
ここで，事前に学習された規制の特徴量は実際に異なるタスクでどの程度機能するかという疑問は生じる  



![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/12.png)    
![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/13.png)  
![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/14.png)  

Performance of off-the-shelf pre-trained models vs. specialized task-focused deep learning models  

事前にトレーニングされたモデルの昨日が非常に特殊なタスクに焦点を当てた深層学習モデルよりも一貫して優れていることが明確にわかる．  

### Fine Tuning Off-the-shelf Pre-trained Models  

これは最終層(分類や回帰)を置き換えるだけではなく，前の層のいくつかを選択的に再学習する問手法．  
DNNは様々なハイパーパラメータをもつ高度に構成可能なアーキテクチャである．  
次の図は顔認証問題の例で，ネットワークの初期の階層が非常に一般的な特徴を学習し，上位層が非常にタスクに特化した特徴を学習していることを示唆する．  


![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/15.png)  


この知識を利用して再トレーニング中に特定の層を固定したり残りの層を必要に応じて微調整したりすることができる．  
この場合ネットワークの全体的なアーキテクチャに関する知識を活用し，その状態を再学習ステップの出発点として使用する．  
これにより，少ない学習時間でより良いパフォーマンスを得ることができる．  

![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/16.png)  

## Freezing or Fine-tuning?


ネットワーク内のレイヤーをフリーズして昨日抽出昨日として使用する必用があるのか，それともプロセス内のレイヤーを微調整する必用があるのかという疑問が生じる  
![](https://github.com/nissy-shota/Transfer-Learning/blob/main/Comprehensive-Hands-on-Guide/images/17.png)  

## Pre-trained Models

転移学習の基本的な要件の一つはソースタスクで優れた性能を発揮するモデルの存在である．  
事前学習モデルは通常安定した状態まで学習されたモデルが達成した数百万のパラメータとウェイトの形で共有される．  

- VGG-16  
- VGG-19  
- Inception V3  
- XCeption  
- ResNet-50  

cv分野では以上のモデルがよく利用される．  

- Word2Vec  
- GloVe  
- FastText  
 
nlpタスクの場合性質が多様であるため，少し困難であるが，embedding modelを使用できる．   

最近では，以下の進捗があった．  
- Universal Sentence Encoder by Google  
- Bidirectional Encoder Representations from Transformers (BERT) by Google  


## Types of Deep Transfer Learning

転移学習に関する文献は多くの繰返しを求めており，
本性の冒頭に述べたように用語はゆるく，しばしば交換可能に使用されている．
そのため，transfer learning, domain adaptation, multitask learningをくべつすることは，
ときに混乱を招く．

これらはすべて関連しており，似たような問題を解決しよとするもの．
一般的には転移学習は概念や原則として考えるべきで，元のタスク-ドメイン知識を使ってターゲットタスクを解決するものである．

### Domain Adaptation

ドメイン適応とは，通常ソースドメインとターゲットドメインのmarginal probabuilityがP{Xs} neq p{Xt}のように異なる場合に言及される．
ソースドメインとターゲットドメインのデータ分布には固定のシフトやドリフトがあり，学習を移すためには微調整が必用である．
例えば，ポジティブまたはネガティブにラベル老けされた映画のレビューのコーパスは製品レビューの感情のコーパスとは異なる．
映画レビューのセンチメントで訓練された，分類機が製品レビューの分類に使用された場合異なる分布となる．
このようなシナリオではDomain Adaptationが使用される．

### Domain Confusion

私たちは、さまざまなtransfer learning戦略を学び、さらに、ソースからターゲットに知識を伝達するために、何を、いつ、どのようにするかという3つの質問について議論しました。
特に、特徴表現の転送がどのように役立つかについて説明しました。繰り返しになりますが、深層学習ネットワークの異なる層は、異なる特徴のセットをキャプチャします。
この事実を利用して、ドメイン不変の特徴を学習し、ドメイン間の転送性を向上させることができます。
モデルに任意の表現を学習させるのではなく、両ドメインの表現ができるだけ似たものになるように誘導する。
これは、ある種の前処理ステップを表現そのものに直接適用することで達成できる。
これらの方法は、Baochen Sun、Jiashi Feng、Kate Saenkoの論文
「Return of Frustratingly Easy Domain Adaptation」でも紹介されています。
この表現の類似性への後押しは、Ganinらの論文「Domain-Adversarial Training of Neural Networks」でも紹介されている。
この手法の基本的な考え方は、ソースモデルに別の目的を追加して、ドメイン自体を混乱させることで類似性を促す、つまりドメインコンフュージョンです。

### Multitask Learning



マルチタスク学習は、転移学習の世界でも少し変わった趣向を凝らしています。
マルチタスク学習では、複数のタスクをソースとターゲットの区別なく同時に学習する。
この場合、学習者は複数のタスクに関する情報を一度に受け取ることになりますが、転移学習の場合は、学習者は最初はターゲットとなるタスクについて何も知りません。
これを図で表すと次のようになる。


### One-shot Learning


深層学習システムは、もともとデータ量が多く、重みを学習するためには多くの学習例が必要となります。
これが深層ニューラルネットワークの限界でもあるのですが、人間の学習はそうではありません。
例えば、子供はリンゴの形を見せられれば、（1つまたは数個の学習例で）簡単に別の品種のリンゴを識別することができますが、MLやディープラーニングのアルゴリズムではそのようなことはありません。
ワンショット学習は、伝達学習の変形で、たった1つまたは数個の学習例に基づいて必要な出力を推論しようとするものです。
これは、（分類タスクであれば）すべての可能なクラスのラベル付きデータを持つことができない実世界のシナリオや、新しいクラスが頻繁に追加される可能性のあるシナリオにおいて、本質的に有用です。
ワンショット学習という言葉が生まれたのも、この分野の研究が始まったのも、Fei-Feiとその共同研究者による画期的な論文「One Shot Learning of Object Categories」がきっかけだと言われています。
この論文では、物体分類のための表現学習に関するベイズの枠組みのバリエーションを提示しました。このアプローチはその後改良され、深層学習システムを用いて応用されています。


### Zero-shot Learning

 ゼロショット学習は、ラベル付けされた例を一切使用せずにタスクを学習する、転移学習のもう一つの極端な変種です。 
 例を使った学習がほとんどの教師付き学習アルゴリズムの目的であるにもかかわらず、これは信じられないことのように聞こえるかもしれません。 
 ゼロデータ学習またはゼロショート学習法は、見たことのないデータを理解するために追加の情報を利用するために、学習段階自体で巧妙な調整を行います。 
 Goodfellowとその共同執筆者は、Deep Learningに関する著書の中で、
 ゼロショット学習を、従来の入力変数x、従来の出力変数y、タスクを記述する追加のランダム変数Tのような3つの変数を学習するシナリオとして提示しています.
 
 
