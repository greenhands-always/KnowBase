-- 数据库初始化脚本
-- 执行顺序：先执行此文件，再根据需要选择schema版本

-- 1. 创建数据库（如果需要）
-- CREATE DATABASE ai_trend_summary;
-- \c ai_trend_summary;

-- 2. 创建必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- 用于模糊匹配
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- 用于复合索引优化

-- 3. 创建基础枚举类型（schema_v2需要）
DO $$
BEGIN
    -- 检查枚举类型是否存在，不存在则创建
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'acquisition_method') THEN
        CREATE TYPE acquisition_method AS ENUM (
            'batch_crawl',      -- 批量爬取
            'stream_crawl',     -- 实时流爬取
            'user_import',      -- 用户导入
            'api_sync',         -- API同步
            'manual_entry'      -- 手动录入
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'processing_stage') THEN
        CREATE TYPE processing_stage AS ENUM (
            'ingested',         -- 已摄入
            'deduplicated',     -- 已去重
            'preprocessed',     -- 已预处理
            'content_extracted', -- 内容提取完成
            'concepts_extracted', -- 概念提取完成
            'classified',       -- 已分类
            'relations_extracted', -- 关系提取完成
            'summarized',       -- 已总结
            'quality_assessed', -- 质量评估完成
            'indexed',          -- 已索引
            'completed'         -- 完全处理完成
        );
    END IF;
END $$;

-- 4. 创建基础用户表（schema_v2需要引用）
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255),
    full_name VARCHAR(200),
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 创建基础的domains表（如果schema.sql中有定义）
CREATE TABLE IF NOT EXISTS domains (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    parent_id INTEGER REFERENCES domains(id),
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. 创建基础的article_types表
CREATE TABLE IF NOT EXISTS article_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 创建基础的concepts表
CREATE TABLE IF NOT EXISTS concepts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    normalized_name VARCHAR(200) NOT NULL,
    type VARCHAR(50),
    description TEXT,
    aliases JSONB,
    frequency INTEGER DEFAULT 1,
    importance_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(normalized_name, type)
);

-- 提示信息
SELECT 'Database setup completed. Now you can run either schema.sql or schema_v2.sql' as message;


-- 增强版数据库设计 - 支持多种数据获取方式和智能重复检测
-- 版本: 2.0.0
-- 新增功能: 数据来源跟踪、重复检测、状态管理优化

-- =============================================================================
-- 1. 数据获取方式和来源管理
-- =============================================================================


-- 数据来源配置表 (增强版)
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL, -- 'blog', 'twitter', 'paper', 'news', 'forum', 'user_import'
    acquisition_method acquisition_method NOT NULL,
    base_url VARCHAR(500),
    description TEXT,
    is_active BOOLEAN DEFAULT true,

    -- 爬取配置
    crawl_frequency_hours INTEGER DEFAULT 24,
    last_crawl_time TIMESTAMP,
    next_crawl_time TIMESTAMP,

    -- 流式配置
    stream_config JSONB, -- 流式爬取的配置参数

    -- 用户导入配置
    import_config JSONB, -- 用户导入的配置和规则

    config JSONB, -- 其他配置参数
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 2. 重复检测和内容指纹系统
-- =============================================================================

-- 内容指纹表 - 多维度重复检测
CREATE TABLE content_fingerprints (
    id SERIAL PRIMARY KEY,

    -- 基础指纹
    content_hash VARCHAR(64) NOT NULL, -- 全文内容SHA256
    title_hash VARCHAR(64), -- 标题hash
    url_hash VARCHAR(64), -- URL标准化后的hash

    -- 语义指纹
    semantic_hash VARCHAR(64), -- 基于语义的hash (可选)
    key_sentences_hash VARCHAR(64), -- 关键句子的hash

    -- 结构化指纹
    author_title_hash VARCHAR(64), -- 作者+标题的组合hash
    domain_content_hash VARCHAR(64), -- 域名+核心内容的hash

    -- 相似度检测
    simhash BIGINT, -- SimHash用于近似重复检测
    minhash JSONB, -- MinHash数组用于Jaccard相似度

    -- 元数据
    word_count INTEGER,
    char_count INTEGER,
    language VARCHAR(10),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 确保内容hash唯一
    UNIQUE(content_hash)
);

-- 重复检测规则表
CREATE TABLE deduplication_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,

    -- 检测策略
    strategy VARCHAR(50) NOT NULL, -- 'exact_match', 'fuzzy_match', 'semantic_match', 'hybrid'

    -- 匹配阈值
    similarity_threshold DECIMAL(3,2) DEFAULT 0.85, -- 相似度阈值

    -- 匹配字段权重
    field_weights JSONB, -- 各字段的权重配置

    -- 规则配置
    config JSONB,

    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 5, -- 规则优先级
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 3. 增强的文章表
-- =============================================================================

-- 文章主表 (增强版)
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,

    -- 基础信息
    source_id INTEGER REFERENCES data_sources(id),
    external_id VARCHAR(200), -- 外部平台的ID
    title TEXT NOT NULL,
    url VARCHAR(1000),
    author VARCHAR(200),
    published_at TIMESTAMP,

    -- 获取信息
    acquisition_method acquisition_method NOT NULL,
    acquisition_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acquisition_metadata JSONB, -- 获取时的元数据

    -- 重复检测
    fingerprint_id INTEGER REFERENCES content_fingerprints(id),
    is_duplicate BOOLEAN DEFAULT false,
    duplicate_of INTEGER REFERENCES articles(id), -- 指向原始文章
    duplicate_confidence DECIMAL(3,2), -- 重复检测置信度

    -- 内容信息
    summary TEXT, -- AI生成的摘要
    language VARCHAR(10) DEFAULT 'en',
    word_count INTEGER,
    reading_time_minutes INTEGER,

    -- 评分系统
    quality_score DECIMAL(3,2), -- 0.00-1.00 质量评分
    importance_score DECIMAL(3,2), -- 0.00-1.00 重要性评分
    trending_score DECIMAL(5,2), -- 热度评分
    novelty_score DECIMAL(3,2), -- 新颖性评分 (基于重复检测)

    -- 存储引用
    raw_content_mongo_id VARCHAR(100), -- MongoDB中原始内容的ID

    -- 用户导入特殊字段
    imported_by INTEGER REFERENCES users(id), -- 导入用户
    import_source VARCHAR(200), -- 导入来源描述
    import_batch_id VARCHAR(100), -- 批次ID

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 约束
    UNIQUE(source_id, external_id),

    -- 检查约束
    CONSTRAINT chk_duplicate_logic CHECK (
        (is_duplicate = false AND duplicate_of IS NULL) OR
        (is_duplicate = true AND duplicate_of IS NOT NULL AND duplicate_of != id)
    )
);


-- 处理状态表 (增强版)
CREATE TABLE processing_status (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    stage processing_stage NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'skipped'

    -- 处理详情
    processor_name VARCHAR(100), -- 处理器名称
    processor_version VARCHAR(50), -- 处理器版本
    processing_time_seconds INTEGER,

    -- 结果信息
    result_data JSONB, -- 处理结果数据
    confidence_score DECIMAL(3,2), -- 处理结果置信度

    -- 错误处理
    error_message TEXT,
    error_code VARCHAR(50),
    retry_count INTEGER DEFAULT 0,

    -- 依赖关系
    depends_on_stages processing_stage[], -- 依赖的前置阶段

    -- 时间戳
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(article_id, stage)
);

-- 处理任务队列表 (增强版)
CREATE TABLE processing_queue (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    stage processing_stage NOT NULL,
    task_type VARCHAR(50) NOT NULL, -- 具体任务类型

    -- 优先级和调度
    priority INTEGER DEFAULT 5, -- 1-10, 1最高优先级
    scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 计划执行时间

    -- 任务配置
    parameters JSONB,
    timeout_seconds INTEGER DEFAULT 300, -- 超时时间

    -- 状态管理
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'cancelled'
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- 执行信息
    worker_id VARCHAR(100), -- 执行的worker标识
    error_message TEXT,

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP

);
CREATE INDEX idx_queue_status_priority ON processing_queue(status, priority DESC, scheduled_at);
CREATE INDEX idx_queue_article_stage ON processing_queue(article_id, stage);
-- =============================================================================
-- 5. 重复检测和复用机制
-- =============================================================================

-- 重复检测结果表
CREATE TABLE duplicate_detection_results (
    id SERIAL PRIMARY KEY,

    -- 检测对象
    source_article_id INTEGER REFERENCES articles(id),
    target_article_id INTEGER REFERENCES articles(id),

    -- 检测方法和结果
    detection_method VARCHAR(50), -- 'content_hash', 'title_similarity', 'semantic_similarity'
    similarity_score DECIMAL(5,4), -- 相似度分数
    detection_rule_id INTEGER REFERENCES deduplication_rules(id),

    -- 匹配详情
    matched_fields JSONB, -- 匹配的字段详情
    confidence_level VARCHAR(20), -- 'high', 'medium', 'low'

    -- 处理决策
    action_taken VARCHAR(50), -- 'mark_duplicate', 'merge', 'keep_both', 'manual_review'
    reviewed_by INTEGER REFERENCES users(id),
    review_notes TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 处理结果复用表
CREATE TABLE processing_reuse_log (
    id SERIAL PRIMARY KEY,

    -- 复用关系
    source_article_id INTEGER REFERENCES articles(id), -- 被复用的文章
    target_article_id INTEGER REFERENCES articles(id), -- 复用到的文章

    -- 复用的处理阶段
    reused_stages processing_stage[],

    -- 复用策略
    reuse_strategy VARCHAR(50), -- 'full_copy', 'partial_copy', 'reference_only'
    adaptation_rules JSONB, -- 适配规则

    -- 质量评估
    reuse_confidence DECIMAL(3,2),
    validation_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'validated', 'rejected'

    -- 元数据
    reused_by VARCHAR(100), -- 执行复用的系统/用户
    reuse_reason TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 6. 用户导入增强支持
-- =============================================================================

-- 用户导入批次表
CREATE TABLE import_batches (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100) UNIQUE NOT NULL,

    -- 导入信息
    imported_by INTEGER REFERENCES users(id),
    import_source VARCHAR(200), -- 'file_upload', 'api_import', 'manual_entry'
    source_description TEXT,

    -- 批次统计
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    successful_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    duplicate_items INTEGER DEFAULT 0,

    -- 处理配置
    processing_config JSONB,
    deduplication_enabled BOOLEAN DEFAULT true,
    auto_process BOOLEAN DEFAULT true,

    -- 状态
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- 导入项目详情表
CREATE TABLE import_items (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100) REFERENCES import_batches(batch_id),

    -- 原始数据
    raw_data JSONB, -- 用户提供的原始数据
    normalized_data JSONB, -- 标准化后的数据

    -- 处理结果
    article_id INTEGER REFERENCES articles(id), -- 关联的文章ID
    processing_status VARCHAR(20) DEFAULT 'pending',

    -- 重复检测结果
    duplicate_detected BOOLEAN DEFAULT false,
    duplicate_article_id INTEGER REFERENCES articles(id),
    duplicate_confidence DECIMAL(3,2),

    -- 错误信息
    error_message TEXT,
    validation_errors JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 7. 索引优化
-- =============================================================================

-- 文章表索引
CREATE INDEX idx_articles_acquisition_method ON articles(acquisition_method);
CREATE INDEX idx_articles_fingerprint ON articles(fingerprint_id);
CREATE INDEX idx_articles_duplicate ON articles(is_duplicate, duplicate_of);
CREATE INDEX idx_articles_source_published ON articles(source_id, published_at DESC);
CREATE INDEX idx_articles_import_batch ON articles(import_batch_id);

-- 内容指纹表索引
CREATE INDEX idx_fingerprints_content_hash ON content_fingerprints(content_hash);
CREATE INDEX idx_fingerprints_title_hash ON content_fingerprints(title_hash);
CREATE INDEX idx_fingerprints_simhash ON content_fingerprints(simhash);

-- 处理状态表索引
CREATE INDEX idx_processing_status_article_stage ON processing_status(article_id, stage);
CREATE INDEX idx_processing_status_stage_status ON processing_status(stage, status);

-- 重复检测结果索引
CREATE INDEX idx_duplicate_results_source ON duplicate_detection_results(source_article_id);
CREATE INDEX idx_duplicate_results_similarity ON duplicate_detection_results(similarity_score DESC);

-- =============================================================================
-- 8. 触发器和自动化
-- =============================================================================

-- 自动更新updated_at字段的触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为相关表添加触发器
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 自动创建处理任务的触发器函数
CREATE OR REPLACE FUNCTION create_initial_processing_tasks()
RETURNS TRIGGER AS $$
BEGIN
    -- 为新文章创建初始处理任务
    IF NEW.is_duplicate = false THEN
        INSERT INTO processing_queue (article_id, stage, task_type, priority)
        VALUES
            (NEW.id, 'deduplicated', 'duplicate_check', 1),
            (NEW.id, 'preprocessed', 'content_preprocessing', 3),
            (NEW.id, 'content_extracted', 'content_extraction', 4),
            (NEW.id, 'concepts_extracted', 'concept_extraction', 5);
    END IF;

    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER create_processing_tasks_on_insert AFTER INSERT ON articles
    FOR EACH ROW EXECUTE FUNCTION create_initial_processing_tasks();