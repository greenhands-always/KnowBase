# 使用文件夹导入器
from modules.knowbase.core.importer import ImporterFactory, ImporterType
from infrastructure.utils.db.DatabaseManager import DatabaseManager, AcquisitionMethod

# 创建数据库管理器
db_manager = DatabaseManager()

# 创建文件夹导入器
importer = ImporterFactory.create_importer(
    ImporterType.FOLDER,
    database_manager=db_manager,
    data_source_name="my_documents",
    acquisition_method=AcquisitionMethod.USER_IMPORT,
    recursive=True,
    auto_process=True,
    deduplication_enabled=True
)

# 执行导入
result = importer.import_data("/path/to/documents", user_id=1)

print(f"导入完成: {result.imported_items}/{result.total_items} 成功")
print(f"重复项: {result.duplicate_items}")
print(f"失败项: {result.failed_items}")
if result.errors:
    print(f"错误: {result.errors}")