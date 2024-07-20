int main() {
    FILE *file = fopen("input.txt", "r");
    if (!file) {
        perror("Failed to open file");
        return 1;
    }

    ASTNode* ast = parse(file);
    fclose(file);

    printf("; Assembly code generated from SimpleLang\n");
    generateAssembly(ast);

    return 0;
}
