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






















