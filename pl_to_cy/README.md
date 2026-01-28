# Problog (RDF) to Neo4j Cypher

这是一个轻量解析器，用于将满足 RDF 约定的 Problog 事实转换为 Neo4j Cypher：

- **一元谓词**：作为实体，谓词名为类型标签，参数为实例。
- **二元谓词**：作为关系边，谓词名为关系名，第一个参数为主语实例，第二个参数为宾语实例。

## 使用方法

```bash
python pl_to_cy.py input.pl -o output.cypher
```

若不指定 `-o` 则输出到标准输出。

## 运行 Cypher（neo4jrunner）

```bash
python neo4jrunner.py output.cypher --uri bolt://localhost:7687 --user neo4j --password neo4j
```

也可使用环境变量 `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`。运行前请确保已安装 `neo4j` Python 驱动。

## 输入示例

```
person(alice).
city(beijing).
lives_in(alice, beijing).
```

## 输出示例

```
MERGE (n:Entity {id: 'alice'})
SET n:`person`
MERGE (n:Entity {id: 'beijing'})
SET n:`city`
MERGE (s:Entity {id: 'alice'})
MERGE (o:Entity {id: 'beijing'})
MERGE (s)-[:`lives_in`]->(o)
```

## 说明

- 解析器忽略 `%` 行内注释。
- 仅支持一元或二元谓词。
- 标签与关系名会自动使用反引号包裹以避免 Cypher 关键字冲突。
