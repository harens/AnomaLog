# Parsers

Parsers are responsible for two distinct stages:

- turning raw log lines into structured records
- turning structured message text into templates

This page is the reference for both the built-in parser implementations and the
protocols they satisfy.

```pycon
>>> from anomalog.parsers import BGLParser, Drain3Parser, IdentityTemplateParser
>>> BGLParser.name
'bgl'
>>> Drain3Parser.name
'drain3'
>>> IdentityTemplateParser("demo").inference("node 7 failed")
('node 7 failed', [])
```

## `anomalog.parsers`

::: anomalog.parsers
