---
name: naming-conventions
description: Naming convention for this project - use when writing or reviewing any Python code (variables, functions, methods, classes) in this repo.
---

# Naming conventions

All identifiers in this project use **camelCase**, except global variables, which follow the **market standard** (PEP8) convention instead.

- Local variables: camelCase (e.g. `userCount`, `responseData`)
- Function names: camelCase (e.g. `getUser`, `parseRequest`)
- Method names: camelCase (e.g. `def fetchItem(self):`)
- Parameters: camelCase (e.g. `def createUser(userName, isActive):`)
- Class names: PascalCase (e.g. `UserProfile`) - standard even in camelCase-style codebases
- Global / module-level variables and constants: market-standard PEP8 style, i.e. `UPPER_SNAKE_CASE` (e.g. `MAX_RETRIES`, `DEFAULT_TIMEOUT`)

Do not use snake_case for anything except global constants/variables. Do not mix styles within the same category.
