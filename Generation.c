void generateAssembly(ASTNode* node) {
    if (!node) return;
    switch (node->type) {
        case NODE_VAR_DECL:
            printf("; Declare variable %s\n", node->text);
            break;
        case NODE_ASSIGN:
            printf("; Assign to variable %s\n", node->text);
            generateAssembly(node->left);
            break;
        case NODE_ARITH_OP:
            generateAssembly(node->left);
            generateAssembly(node->right);
            if (strcmp(node->text, "+") == 0) {
                printf("ADD A, B\n");
            } else if (strcmp(node->text, "-") == 0) {
                printf("SUB A, B\n");
            }
            break;
        case NODE_CONDITIONAL:
            printf("; If condition\n");
            generateAssembly(node->left);
            printf("CMP A, B\n");
            printf("JNZ END_IF\n");
            generateAssembly(node->right);
            printf("END_IF:\n");
            break;
    }
}
