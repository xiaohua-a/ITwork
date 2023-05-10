import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns  #使用sns做热力图，观察变量间的相关性
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression

# 读取数据集并添加表头
data = pd.read_csv('招聘数据十万条.csv', header=None, encoding='utf-8')
data.columns = ['爬取时间', '岗位', '公司名称', '所在城市', '月薪', '待遇', '公司性质', '岗位要求']
# 去重
data.drop_duplicates(inplace=True)
# 填充缺失值
data['月薪'].fillna(value=0, inplace=True)
data['待遇'].fillna(value='无', inplace=True)
# 格式化
#将月薪转换为万/月
def unify_salary(data):
    salary = data['月薪']
    if pd.isna(salary):
        return salary
    m1 = re.match(r'(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)千?/月', str(salary))
    m2 = re.match(r'(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)万?/月', str(salary))
    m3 = re.match(r'(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)万?/年', str(salary))
    if m1:
        low = float(m1.group(1)) / 10
        high = float(m1.group(2)) / 10
        return '{:.2f}-{:.2f}万/月'.format(low, high)
    elif m2:
        return salary.replace('千/月', '万/月')
    elif m3:
        low = float(m3.group(1)) / 12
        high = float(m3.group(2)) / 12
        return '{:.2f}-{:.2f}万/月'.format(low, high)
    else:
        return ''
data['月薪'] = data.apply(unify_salary, axis=1)
data['月薪']=data['月薪'].apply(lambda x: x.split('-', 1)[0] if '-' in str(x) else x )
data['公司性质'] = data['公司性质'].apply(lambda x: x if x in ['国企', '民营公司', '上市公司', 
    '合资', '外资（欧美）', '外资（非欧美）'] else '其他')
data[['工作地', '工作经验', '学历']] = data['岗位要求'].str.split(' ', expand=True)
# 提取城市信息
data['工作地'] = data['工作地'].str[:2]
data['工作经验'].fillna(value='无需经验', inplace=True)
#data['工作经验']=data['工作经验'].apply(lambda x: x.split('-', 1)[0] if '-' in str(x) else x )
#处理工作年限为数字
def unify_experience(data):
    experience = data['工作经验']
    if pd.isna(experience):
        return experience
    if experience == '无需经验':
        return experience
    else:
        nums = re.findall(r'\d+', experience)
        if len(nums) > 0:
            return nums[0]
        else:
            return ''
data['工作经验'] = data.apply(unify_experience, axis=1)
data['学历'].fillna(value='不限', inplace=True)
# 计算冗余字段
data.drop(columns=['爬取时间','岗位要求','所在城市'], inplace=True)
# 保存清洗后的数据
data.to_csv('招聘数据清洗后.csv', index=False, encoding='utf-8')

# 读取CSV文件
df = pd.read_csv('招聘数据清洗后.csv')
# 计算招聘企业数
company_count = df['公司名称'].nunique()
# 计算岗位数
position_count = df['岗位'].nunique()
# 计算平均月薪
mean_salary = df['月薪'].mean()
# 将结果保存为CSV文件
result = pd.DataFrame({'招聘企业数': [company_count], '岗位数': [position_count], '平均月薪': [mean_salary]})
result.to_csv('IT行业招聘整体情况.csv', index=False)
# 工作地分布
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 6))
sns.countplot(x='工作地', data=df, order=df['工作地'].value_counts().index[:20])
plt.title('工作地分布', fontsize=20)
plt.xlabel('工作地', fontsize=16)
plt.ylabel('数量', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.show()

# 月薪分布
# 计算每个地区的平均月薪
df['月薪'] = df['月薪'].astype(float)
avg_salary = df.groupby('工作地')['月薪'].mean().reset_index()[:20]
# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(avg_salary['工作地'], avg_salary['月薪'], marker='o')
plt.title('前20个地区的平均月薪变化', fontsize=20)
plt.xlabel('工作地', fontsize=16)
plt.ylabel('平均月薪（万/月）', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.show()

# 薪资最高的TOP10
salary_top10 = df[['岗位', '公司名称', '月薪']].sort_values(by='月薪', ascending=False)[:20]
plt.figure(figsize=(8, 6))
sns.barplot(x='月薪', y='岗位', data=salary_top10)
plt.title('薪资最高的TOP20', fontsize=20)
plt.xlabel('月薪（万/月）', fontsize=16)
plt.ylabel('', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# 工作经验分布
plt.figure(figsize=(8, 6))
sns.countplot(x='工作经验', data=df, order=df['工作经验'].value_counts().index)
plt.title('工作经验分布', fontsize=20)
plt.xlabel('工作经验（年）', fontsize=16)
plt.ylabel('数量', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.show()

# 学历要求分布
plt.figure(figsize=(8, 6))
sns.countplot(x='学历', data=df, order=df['学历'].value_counts().index)
plt.title('学历要求分布', fontsize=20)
plt.xlabel('学历要求', fontsize=16)
plt.ylabel('数量', fontsize=16)
plt.xticks(fontsize=12)
plt.show()

# 公司性质分布
#plt.figure(figsize=(8, 6))
#sns.countplot(x='公司性质', data=df, order=df['公司性质'].value_counts().index)
#plt.title('公司性质分布', fontsize=20)
#plt.xlabel('公司性质', fontsize=16)
#plt.ylabel('数量', fontsize=16)
#plt.xticks(fontsize=12)
#plt.show()

# 统计各个公司性质的数量
counts = df['公司性质'].value_counts()
# 绘制饼图
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('公司性质分布', fontsize=20)
plt.show()

# 将待遇列中有待遇的值设为1，无待遇的值设为0
df['待遇'] = df['待遇'].apply(lambda x: 0 if x == '无' else 1)
# 绘制饼图
plt.figure(figsize=(6, 6))
labels = ['无待遇', '有待遇']
sizes = [df['待遇'].value_counts()[0], df['待遇'].value_counts()[1]]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('待遇占比', fontsize=20)
plt.show()

# 定义待遇列表，用于存储待遇中的有意义的词语
#treatments = ['五险一金', '定期体检', '绩效奖金', '带薪假期', '周末双休', '交通补贴', '餐饮补贴', '年终奖金', '弹性工作', '加班补助', '节日福利', '岗位晋升', '团建旅游', '免费培训', '全勤奖', '员工旅游', '年度旅游', '包吃住', '话费补贴', '定期体检']
# 将待遇文本进行预处理，保留一些有意义的词语
#text = ' '.join(df[df['待遇'] == 1]['待遇'].apply(lambda x: ' '.join(filter(lambda t: t in x, treatments))))
# 绘制词云图
#wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.title('频率最高的前10个待遇', fontsize=20)
#plt.show()
# 按照工作地和岗位分组，求平均月薪
grouped = df.groupby(['工作地', '岗位'])['月薪'].mean()[:10]
# 将分组结果转化为DataFrame
result = grouped.unstack()
# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(result, cmap='YlGnBu', annot=True, fmt='.1f')
plt.title('不同岗位在不同城市的平均月薪', fontsize=20)
plt.xlabel('岗位', fontsize=16)
plt.ylabel('工作地', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
# 对数据进行初步的处理和分析
df = df[['公司性质', '月薪']]
df['月薪'] = df['月薪'].astype('float') # 清除月薪单位并转换数据类型
df = df.groupby('公司性质').mean().reset_index()
# 用柱状图对公司性质和平均月薪进行可视化
plt.bar(df['公司性质'], df['月薪'])
plt.xticks(rotation=45)
plt.xlabel('公司性质')
plt.ylabel('平均月薪')
plt.title('公司性质和平均月薪柱状图')
plt.show()
# 对公司名称进行分组，计算每个公司的平均薪资
grouped = df.groupby('公司名称')['月薪'].mean()
# 对平均薪资进行排序，取出平均薪资最高的20家公司
top_20 = grouped.sort_values(ascending=False)[:20]
# 可视化展示
plt.bar(top_20.index, top_20.values)
plt.xticks(rotation=90)
plt.title('平均薪资最高的20家公司')
plt.xlabel('公司名称')
plt.ylabel('平均月薪')
plt.show()
# 折线图-月薪与工作年限的关系
salary_by_exp = df.groupby('工作经验')['月薪'].mean()
plt.plot(salary_by_exp.index, salary_by_exp.values)
plt.xlabel('工作经验')
plt.ylabel('月薪')
plt.title('月薪与工作经验的关系')
plt.show()
# 热力图-不同城市、不同工作经验、不同学历的平均月薪
salary_pivot = pd.pivot_table(df, index='工作地', columns=['工作经验', '学历'], values='月薪', aggfunc='mean')[:10]
plt.imshow(salary_pivot, cmap='YlOrRd', interpolation='nearest')
plt.xticks(range(len(salary_pivot.columns[:10])), salary_pivot.columns, rotation=90)
plt.yticks(range(len(salary_pivot.index[:10])), salary_pivot.index)
plt.colorbar()
plt.title('不同城市、不同工作经验、不同学历的平均月薪')
plt.show()
# 对学历进行分组，计算每个学历的平均薪资
grouped = df.groupby('学历')['月薪'].mean()
# 可视化展示
plt.plot(grouped.index, grouped.values)
plt.title('平均薪资和学历的关系')
plt.xlabel('学历')
plt.ylabel('平均月薪')
plt.show()

'''#模型预测
def predict(data, education):
    """
    :param data: 训练数据
    :param education: 学历
    :return: 模型得分，10年工作预测
    """
    train = data[data['education'] == education].to_numpy()
    x = train[:, 1:2]
    y = train[:, 2]

    # model 训练
    model = LinearRegression()
    model.fit(x, y)

    # model 预测
    X = [[i] for i in range(11)]
    return model.score(x, y), model.predict(X)

education_list = ['小学', '初中', '中专', '高中', '大专', '本科', '硕士', '博士']
data = pd.read_csv('招聘数据清洗后.csv')
scores, values = [], []
for education in education_list:
    score, y = predict(data, education)
    scores.append(score)
    values.append(y)

result = pd.DataFrame()
result['学历'] = education_list
result['模型得分'] = scores
result['(1年经验)平均工资'] = [value[1] for value in values]
result['(3年经验)平均工资'] = [value[2] for value in values]
result['(5年经验)平均工资'] = [value[4] for value in values]
result['(10年经验)平均工资'] = [value[10] for value in values]
print(result)
'''