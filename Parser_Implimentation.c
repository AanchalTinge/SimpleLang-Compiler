Token currentToken;

void advanceToken(FILE *file) {
    getNextToken(file, &currentToken);
}

ASTNode* parseExpression(FILE *file);

ASTNode* parsePrimary(FILE *file) {
    ASTNode* node = NULL;
    if (currentToken.type == TOKEN_IDENTIFIER || currentToken.type == TOKEN_NUMBER) {
        node = createNode(NODE_ARITH_OP, currentToken.text);
        advanceToken(file);
    }
    return node;
}

ASTNode* parseExpression(FILE *file) {
    ASTNode* left = parsePrimary(file);
    while (currentToken.type == TOKEN_PLUS || currentToken.type == TOKEN_MINUS) {
        ASTNode* node = createNode(NODE_ARITH_OP, currentToken.text);
        advanceToken(file);
        node->left = left;
        node->right = parsePrimary(file);
        left = node;
    }
    return left;
}

ASTNode* parseStatement(FILE *file) {
    ASTNode* node = NULL;
    if (currentToken.type == TOKEN_INT) {
        advanceToken(file);
        if (currentToken.type == TOKEN_IDENTIFIER) {
            node = createNode(NODE_VAR_DECL, currentToken.text);
            advanceToken(file);
            if (currentToken.type == TOKEN_SEMICOLON) {
                advanceToken(file);
            }
        }
    } else if (currentToken.type == TOKEN_IDENTIFIER) {
        node = createNode(NODE_ASSIGN, currentToken.text);
        advanceToken(file);
        if (currentToken.type == TOKEN_ASSIGN) {
            advanceToken(file);
            node->left = parseExpression(file);
            if (currentToken.type == TOKEN_SEMICOLON) {
                advanceToken(file);
            }
        }
    } else if (currentToken.type == TOKEN_IF) {
        advanceToken(file);
        if (currentToken.type == TOKEN_LPAREN) {
            advanceToken(file);
            node = createNode(NODE_CONDITIONAL, "if");
            node->left = parseExpression(file);
            if (currentToken.type == TOKEN_RPAREN) {
                advanceToken(file);
                if (currentToken.type == TOKEN_LBRACE) {
                    advanceToken(file);
                    node->right = parseStatement(file);
                    if (currentToken.type == TOKEN_RBRACE) {
                        advanceToken(file);
                    }
                }
            }
        }
    }
    return node;
}

ASTNode* parse(FILE *file) {
    advanceToken(file);
    ASTNode* root = parseStatement(file);
    return root;
}
